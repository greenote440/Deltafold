"""
TCPNet adapter — bridges the ZW471/TopoteinWorkshop equivariant `tcpnet_v1`
encoder into the Deltafold contrastive training pipeline behind `--model topotein`.

Why an adapter is needed
------------------------
`topotein.models.graph_encoders.tcpnet_v1.TCPNetModel` is an SE(3)-equivariant
GCP/ScalarVector model built on the ProteinWorkshop (PyG + graphein + Hydra)
framework. It consumes a graphein ``ProteinBatch`` carrying full-backbone
geometry, a topotein SSE *cell complex*, and rank-0/1/2/3 scalar+vector features
produced by ``topotein.features.factory.TopoteinFeaturiser``. The Deltafold
pipeline, by contrast, feeds a hand-rolled ``rank0/rank1/rank2/rank3`` dict that
is CA-only and cannot drive pydssp SSE assignment (which needs N,CA,C,O).

So this adapter rebuilds the geometry from the *raw PDB structures* (streamed
on the fly from ``data/hoan_raw_pdb/virome_pdbs.zip``), featurises them with the
workshop's own ``TopoteinFeaturiser``, runs ``tcpnet_v1``, and projects the
graph embedding to ``scalar_dim`` so it is a drop-in for the contrastive loss.

The model is called by the trainer with the list of sample *paths* for the
batch (the contrastive collate already returns these). Accession ids embedded
in those paths are matched against the PDB zip members.

Run ``python tcpnet_adapter.py`` for a standalone smoke test once the
ProteinWorkshop dependency stack is installed (see README / install notes).
"""
import os
import re
import sys
import zipfile
import tempfile

import torch
import torch.nn as nn

# --- Make the vendored proteinworkshop + topotein packages importable ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXTERNAL = os.path.join(_HERE, "external", "TopoteinWorkshop")


def _make_extension_shims():
    """Inject pure-PyTorch shims for torch_cluster and torch_scatter.

    Both real packages fail to compile on macOS Xcode 16+ against torch ≥2.7
    (libc++ ABI change forbids std::is_arithmetic specialisation).

    IMPORTANT: these shims use ONLY stdlib + torch — no torch_geometric imports.
    They must be injected BEFORE torch_geometric is first imported (torch_geometric
    checks for torch_cluster.knn at import time and replaces the module with a
    raising stub if the check fails).
    """
    import types
    import torch
    import torch.nn.functional as _F

    # ------------------------------------------------------------------ helpers
    def _expand_index(index, src):
        """Expand a 1-D index to match all dimensions of src (dim=0 scatter)."""
        if index.dim() == src.dim():
            return index
        shape = [-1] + [1] * (src.dim() - 1)
        return index.view(shape).expand_as(src)

    def _dim_size(index, dim_size):
        return dim_size if dim_size is not None else (int(index.max().item()) + 1 if index.numel() > 0 else 0)

    # ------------------------------------------------------------------ scatter
    def _scatter_sum(src, index, dim=0, out=None, dim_size=None):
        ds = _dim_size(index, dim_size)
        size = list(src.shape)
        size[dim] = ds
        result = src.new_zeros(size) if out is None else out
        idx = _expand_index(index, src) if dim == 0 else index
        return result.scatter_add_(dim, idx, src)

    def _scatter_mean(src, index, dim=0, out=None, dim_size=None):
        total = _scatter_sum(src, index, dim=dim, dim_size=dim_size)
        count = _scatter_sum(torch.ones_like(src), index, dim=dim, dim_size=dim_size)
        return total / count.clamp(min=1)

    def _scatter_std(src, index, dim=0, out=None, dim_size=None, unbiased=True):
        mean = _scatter_mean(src, index, dim=dim, dim_size=dim_size)
        # `index` may be multi-dimensional (torch_scatter broadcasts it to src's
        # shape), so index_select (1-D only) won't do; expand and gather instead.
        idx_full = _expand_index(index, src)
        mean_exp = torch.gather(mean, dim, idx_full)
        diff_sq = (src - mean_exp) ** 2
        n = _scatter_sum(torch.ones_like(src), index, dim=dim, dim_size=dim_size)
        denom = (n - 1).clamp(min=1) if unbiased else n.clamp(min=1)
        return (_scatter_sum(diff_sq, index, dim=dim, dim_size=dim_size) / denom).sqrt()

    def _scatter_argreduce(src, index, dim, ds, result):
        """Per-segment arg{max,min} index into `src` along `dim`.

        Mirrors torch_scatter's second return value: for each output slot, the
        index along `dim` of the src element that produced the reduced value;
        empty segments get the filler `src.size(dim)`. Callers (e.g. Topotein's
        `localize`) index `scatter_max(...)[1]` directly, so this must NOT be
        None. Ties resolve to the smallest index, which is sufficient here.
        """
        n = src.size(dim)
        shape = [1] * src.dim(); shape[dim] = n
        arange = torch.arange(n, device=src.device).view(shape).expand_as(src)
        gathered = result.index_select(dim, index)
        is_winner = src == gathered
        cand = torch.where(is_winner, arange, torch.full_like(arange, n))
        size = list(src.shape); size[dim] = ds
        arg = src.new_full(size, n, dtype=torch.long)
        idx = _expand_index(index, src) if dim == 0 else index
        return arg.scatter_reduce_(dim, idx, cand, reduce="amin", include_self=True)

    def _scatter_max(src, index, dim=0, out=None, dim_size=None):
        ds = _dim_size(index, dim_size)
        size = list(src.shape); size[dim] = ds
        result = src.new_full(size, float("-inf")) if out is None else out
        idx = _expand_index(index, src) if dim == 0 else index
        result.scatter_reduce_(dim, idx, src, reduce="amax", include_self=True)
        return result, _scatter_argreduce(src, index, dim, ds, result)

    def _scatter_min(src, index, dim=0, out=None, dim_size=None):
        ds = _dim_size(index, dim_size)
        size = list(src.shape); size[dim] = ds
        result = src.new_full(size, float("inf")) if out is None else out
        idx = _expand_index(index, src) if dim == 0 else index
        result.scatter_reduce_(dim, idx, src, reduce="amin", include_self=True)
        return result, _scatter_argreduce(src, index, dim, ds, result)

    def _scatter_softmax(src, index, dim=0, fill_value=0, dim_size=None):
        max_val, _ = _scatter_max(src, index, dim=dim, dim_size=dim_size)
        src_s = src - torch.index_select(max_val, dim, index)
        exp_s = src_s.exp()
        exp_sum = _scatter_sum(exp_s, index, dim=dim, dim_size=dim_size)
        return exp_s / torch.index_select(exp_sum, dim, index).clamp(min=1e-12)

    def _scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
        if reduce in ("sum", "add"):
            return _scatter_sum(src, index, dim, out, dim_size)
        if reduce == "mean":
            return _scatter_mean(src, index, dim, out, dim_size)
        if reduce == "max":
            return _scatter_max(src, index, dim, out, dim_size)[0]
        if reduce == "min":
            return _scatter_min(src, index, dim, out, dim_size)[0]
        raise ValueError(f"Unknown reduce: {reduce}")

    # ------------------------------------------------------------------ cluster
    def _knn(x, y, k, batch_x=None, batch_y=None, cosine=False, num_workers=1):
        # NOTE: this docstring must stay non-empty and must NOT contain the
        # substring 'batch_size'. torch_geometric.typing probes
        # `torch_cluster.knn.__doc__` at import time; a None doc raises a
        # TypeError there, which PyG treats as a failed import and swaps in a
        # raising stub (disabling the shim entirely). Omitting 'batch_size'
        # keeps WITH_TORCH_CLUSTER_BATCH_SIZE=False, matching this signature.
        """k-nearest-neighbor edges (pure-torch shim). Signature mirrors
        torch_cluster.knn(x, y, k, batch_x, batch_y, cosine, num_workers)."""
        if batch_x is None: batch_x = x.new_zeros(x.size(0), dtype=torch.long)
        if batch_y is None: batch_y = y.new_zeros(y.size(0), dtype=torch.long)
        edges = []
        for b in batch_x.unique():
            ix = (batch_x == b).nonzero(as_tuple=True)[0]
            iy = (batch_y == b).nonzero(as_tuple=True)[0]
            xb, yb = (_F.normalize(x[ix], p=2, dim=-1), _F.normalize(y[iy], p=2, dim=-1)) if cosine else (x[ix], y[iy])
            dist = torch.cdist(yb, xb)
            kk = min(k, xb.size(0))
            _, nbr = dist.topk(kk, dim=-1, largest=False)
            row = torch.arange(yb.size(0), device=y.device).unsqueeze(1).expand_as(nbr).reshape(-1)
            edges.append(torch.stack([ix[nbr.reshape(-1)], iy[row]]))
        return torch.cat(edges, dim=1) if edges else x.new_zeros(2, 0, dtype=torch.long)

    def _knn_graph(x, k, batch=None, loop=False, flow="source_to_target", cosine=False, num_workers=1):
        if batch is None: batch = x.new_zeros(x.size(0), dtype=torch.long)
        edges = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=True)[0]
            xb = _F.normalize(x[idx], p=2, dim=-1) if cosine else x[idx]
            dist = torch.cdist(xb, xb)
            if not loop: dist.fill_diagonal_(float("inf"))
            kk = min(k, xb.size(0) - (0 if loop else 1))
            _, nbr = dist.topk(kk, dim=-1, largest=False)
            row = torch.arange(xb.size(0), device=x.device).unsqueeze(1).expand_as(nbr).reshape(-1)
            col = nbr.reshape(-1)
            r, c = idx[row], idx[col]
            edges.append(torch.stack([c, r]) if flow == "source_to_target" else torch.stack([r, c]))
        return torch.cat(edges, dim=1) if edges else x.new_zeros(2, 0, dtype=torch.long)

    def _radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32, num_workers=1):
        if batch_x is None: batch_x = x.new_zeros(x.size(0), dtype=torch.long)
        if batch_y is None: batch_y = y.new_zeros(y.size(0), dtype=torch.long)
        edges = []
        for b in batch_x.unique():
            ix = (batch_x == b).nonzero(as_tuple=True)[0]
            iy = (batch_y == b).nonzero(as_tuple=True)[0]
            dist = torch.cdist(y[iy], x[ix])
            sl, dl = (dist < r).nonzero(as_tuple=True)
            edges.append(torch.stack([ix[dl], iy[sl]]))
        return torch.cat(edges, dim=1) if edges else x.new_zeros(2, 0, dtype=torch.long)

    def _radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target", num_workers=1):
        if batch is None: batch = x.new_zeros(x.size(0), dtype=torch.long)
        edges = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=True)[0]
            dist = torch.cdist(x[idx], x[idx])
            if not loop: dist.fill_diagonal_(float("inf"))
            sl, dl = (dist < r).nonzero(as_tuple=True)
            r2, c2 = idx[sl], idx[dl]
            edges.append(torch.stack([c2, r2]) if flow == "source_to_target" else torch.stack([r2, c2]))
        return torch.cat(edges, dim=1) if edges else x.new_zeros(2, 0, dtype=torch.long)

    def _fps(x, batch=None, ratio=0.5, random_start=True):
        return torch.arange(x.size(0), device=x.device)

    def _graclus_cluster(weight, node_pair_index, num_nodes, num_edges):
        # Stub — returns each node in its own cluster (identity assignment).
        return torch.arange(num_nodes, dtype=torch.long)

    def _grid_cluster(pos, size, start=None, end=None):
        # Stub — torch_geometric.nn.pool.voxel_grid imports this at module load.
        # Identity assignment (each point its own cell); only matters if voxel
        # pooling is actually used, which this model does not do.
        return torch.arange(pos.size(0), dtype=torch.long, device=pos.device)

    def _nearest(x, y, batch_x=None, batch_y=None):
        # Stub — nearest neighbour assignment (single-neighbour knn).
        e = _knn(y, x, 1, batch_x=batch_y, batch_y=batch_x)
        return e[0]

    # ------------------------------------------------------------------ install
    if "torch_cluster" not in sys.modules:
        try:
            import torch_cluster  # noqa: F401
        except ImportError:
            tc = types.ModuleType("torch_cluster")
            tc.knn = _knn
            tc.knn_graph = _knn_graph
            tc.radius = _radius
            tc.radius_graph = _radius_graph
            tc.fps = _fps
            tc.graclus_cluster = _graclus_cluster
            tc.grid_cluster = _grid_cluster
            tc.nearest = _nearest
            sys.modules["torch_cluster"] = tc

    if "torch_scatter" not in sys.modules:
        try:
            import torch_scatter  # noqa: F401
        except ImportError:
            ts = types.ModuleType("torch_scatter")
            ts.scatter = _scatter
            ts.scatter_sum = _scatter_sum
            ts.scatter_add = _scatter_sum
            ts.scatter_mean = _scatter_mean
            ts.scatter_std = _scatter_std
            ts.scatter_max = _scatter_max
            ts.scatter_min = _scatter_min
            ts.scatter_softmax = _scatter_softmax
            sys.modules["torch_scatter"] = ts


# Inject shims at module load time — before any downstream import can trigger
# torch_geometric's typing.py check (which permanently replaces the module with
# a raising stub if torch_cluster.knn is missing at that moment).
_make_extension_shims()


def _ensure_workshop_on_path():
    if _EXTERNAL not in sys.path:
        sys.path.insert(0, _EXTERNAL)
    # proteinworkshop uses pyrootutils / a `.project-root` marker for constants.
    os.environ.setdefault("PROJECT_ROOT", _EXTERNAL)
    # Must run before any torch_geometric import (typing.py checks for knn).
    _make_extension_shims()


_ACC_RE = re.compile(r"([A-Z]{1,2}_[0-9]{5,10})")


def extract_accession(text: str) -> str:
    """Pull the RefSeq-style accession (e.g. YP_010085741) out of a string.

    Mirrors ``train.extract_accession`` so dataset sample paths and PDB zip
    member names key on the same id.
    """
    m = _ACC_RE.search(text)
    return m.group(1) if m else text


class PDBZipIndex:
    """Lazy accession -> member-name index over the raw-PDB zip.

    The archive holds ~132k members in nested family folders; we only build a
    name index once and read individual members on demand (no full unzip).
    """

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self._zf = None
        self._index = None

    @property
    def zf(self) -> zipfile.ZipFile:
        # Open lazily and per-process so DataLoader workers each get a handle.
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    @property
    def index(self) -> dict:
        if self._index is None:
            idx = {}
            for name in self.zf.namelist():
                if not name.lower().endswith(".pdb"):
                    continue
                acc = extract_accession(os.path.basename(name))
                # First occurrence wins; accessions are unique in practice.
                idx.setdefault(acc, name)
            self._index = idx
        return self._index

    def read_pdb_bytes(self, accession: str):
        name = self.index.get(accession)
        if name is None:
            return None
        return self.zf.read(name)


class TCPNetEncoder(nn.Module):
    """Equivariant TCPNet-v1 encoder wrapped to match the Deltafold model API.

    Forward contract (compatible with ``train_contrastive``)::

        z = model(paths=paths)                       # (B, scalar_dim)
        z, h0 = model(paths=paths, return_nodes=True) # (B, dim), (N, dim)

    A ``features=`` positional arg is accepted and ignored for signature
    compatibility with the old ``Topotein``/``AsymmetricTopoNet`` models; this
    encoder rebuilds geometry from ``paths`` instead of the rank-dict.
    """

    # Marker the trainer branches on (the rank-dict models do not set this).
    is_tcpnet = True

    def __init__(
        self,
        scalar_dim: int = 128,
        feature_config: str = "ca_bb_sse",
        encoder_config: str = "tcpnet_v1",
        num_layers: int | None = None,
        pdb_zip: str = os.path.join("data", "hoan_raw_pdb", "virome_pdbs.zip"),
        ignore_featurise_errors: bool = True,
        **_ignored,
    ):
        super().__init__()
        _ensure_workshop_on_path()

        import hydra
        from omegaconf import OmegaConf

        # Register OmegaConf resolvers used by the workshop YAML configs.
        # We do each one individually so a ValueError (already-registered from a
        # previous call in the same process) skips that entry without preventing
        # the others from being registered.  This avoids the silent-swallow bug
        # where catching a broad Exception on the whole block leaves `divide` (or
        # `resolve_feature_config_dim`) unregistered.
        from proteinworkshop.models.utils import get_input_dim

        def _register(name, fn):
            try:
                OmegaConf.register_new_resolver(name, fn)
            except Exception:
                pass  # already registered in this process — fine

        _register("plus", lambda x, y: x + y)
        _register(
            "resolve_feature_config_dim",
            lambda features_config, feature_config_name, task_config, recurse: get_input_dim(
                features_config, feature_config_name, task_config,
                recurse_for_node_features=recurse,
            ),
        )
        _register(
            "resolve_num_edge_types",
            lambda features_config: len(features_config.edge_types),
        )
        _register(
            "divide",
            lambda x, y: (x // y if isinstance(x, int) and isinstance(y, int) else x / y),
        )

        cfg_dir = os.path.join(_EXTERNAL, "proteinworkshop", "config")
        enc = OmegaConf.load(os.path.join(cfg_dir, "encoder", f"{encoder_config}.yaml"))
        feats = OmegaConf.load(os.path.join(cfg_dir, "features", f"{feature_config}.yaml"))

        # `validate_topotein_config`: encoder.features overrides win on the
        # shared feature config (this is how dims stay consistent).
        for key in enc.get("features", {}):
            feats[key] = enc.features[key]
        if num_layers is not None:
            enc.num_layers = num_layers

        root = OmegaConf.create(
            {"encoder": enc, "features": feats, "task": {"task": "none"}}
        )
        OmegaConf.resolve(root)

        self._feature_config = feature_config
        self.featuriser = hydra.utils.instantiate(root.features)
        self.encoder = hydra.utils.instantiate(root.encoder)

        # Graph embedding scalar dim == pr_s_emb_dim (== emb_dim) for tcpnet_v1,
        # but project explicitly so the contrastive head always sees scalar_dim
        # and a normalized output (matching the old Topotein output_head).
        graph_dim = int(root.encoder.model_cfg.p_hidden_dim)
        node_dim = int(root.encoder.model_cfg.h_hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, scalar_dim),
            nn.LayerNorm(scalar_dim),
        )
        self.node_proj = nn.Linear(node_dim, scalar_dim)

        self.pdb_index = PDBZipIndex(pdb_zip)
        self.ignore_featurise_errors = ignore_featurise_errors
        self._scalar_dim = scalar_dim

    _warned_mps = False

    @staticmethod
    def _as_device(a):
        """Return torch.device(a) if `a` names a device, else None."""
        if isinstance(a, torch.device):
            return a
        if isinstance(a, str):
            try:
                return torch.device(a)
            except (RuntimeError, ValueError):
                return None  # a dtype string like "float32", not a device
        return None

    @classmethod
    def _requested_device(cls, args, kwargs):
        """Best-effort extract a target torch.device from .to() arguments."""
        if kwargs.get("device") is not None:
            return torch.device(kwargs["device"])
        for a in args:
            dev = cls._as_device(a)
            if dev is not None:
                return dev
        return None

    def to(self, *args, **kwargs):
        # MPS cannot host torch sparse-COO tensors (no SparseMPS kernel for
        # `_sparse_coo_tensor_with_dims_and_tensors`), and Topotein's cell-complex
        # incidence matrices (N0_2, N2_3) are sparse and indexed against batch.pos
        # in the model's geometry ops — so the whole encoder must share one device
        # and that device can't be MPS. Redirect MPS -> CPU (CUDA is fine).
        dev = self._requested_device(args, kwargs)
        if dev is not None and dev.type == "mps":
            if not TCPNetEncoder._warned_mps:
                print(
                    "[TCPNetEncoder] MPS can't host sparse cell-complex tensors; "
                    "running on CPU instead."
                )
                TCPNetEncoder._warned_mps = True
            kwargs.pop("device", None)
            # Drop any device-naming positional arg; keep dtype/non_blocking.
            args = tuple(a for a in args if self._as_device(a) is None)
            result = super().to("cpu", *args, **kwargs)
        else:
            result = super().to(*args, **kwargs)
        # The featuriser must stay on CPU: pydssp uses float64 (unsupported on
        # MPS) and the batch tensors are CPU-bound until after featurisation.
        self.featuriser = self.featuriser.cpu()
        return result

    # -- geometry reconstruction ------------------------------------------------
    def _load_protein(self, path: str):
        """Stream one structure's PDB from the zip -> graphein Protein object."""
        from graphein.protein.tensor.io import protein_to_pyg

        acc = extract_accession(os.path.basename(path))
        raw = self.pdb_index.read_pdb_bytes(acc)
        if raw is None:
            raise FileNotFoundError(f"No PDB for accession {acc} (from {path})")
        # protein_to_pyg wants a file path; PDBs are small so a temp file is cheap.
        with tempfile.NamedTemporaryFile("wb", suffix=".pdb", delete=False) as fh:
            fh.write(raw)
            tmp = fh.name
        try:
            p = protein_to_pyg(
                path=tmp,
                chain_selection="all",
                keep_insertions=True,
                store_het=False,
            )
        finally:
            os.unlink(tmp)
        # ProteinWorkshop's featuriser expects `seq_pos` (0-indexed residue
        # positions) to already be on the Protein object; `protein_to_pyg`
        # doesn't set it, so add it here.
        if not hasattr(p, "seq_pos") or p.seq_pos is None:
            p.seq_pos = torch.arange(p.coords.shape[0])
        return p

    def _build_batch(self, paths, device):
        from graphein.protein.tensor.data import ProteinBatch

        proteins, keep = [], []
        for i, p in enumerate(paths):
            try:
                proteins.append(self._load_protein(p))
                keep.append(i)
            except Exception as e:
                if not self.ignore_featurise_errors:
                    raise
                print(f"[TCPNetEncoder] skipping {os.path.basename(p)}: {e}")
        if not proteins:
            return None, []
        batch = ProteinBatch.from_protein_list(proteins)
        # graphein's collate sets num_graphs but leaves the node->graph
        # assignment vector (`batch.batch`) as None. Topotein's SSE cell-complex
        # step requires it, so build it from each protein's residue count.
        if getattr(batch, "batch", None) is None:
            counts = torch.tensor([p.coords.shape[0] for p in proteins])
            batch.batch = torch.repeat_interleave(
                torch.arange(len(proteins)), counts
            )
        # Featurise on CPU: pydssp (SSE assignment) uses float64 which MPS
        # doesn't support. Move to the target device only after featurisation.
        batch = self.featuriser(batch)
        batch = batch.to(device)
        return batch, keep

    @staticmethod
    def _as_scalar(rep):
        # tcpnet readout returns a bare tensor when the output vector dim is 0,
        # but fall back to `.scalar` if a ScalarVector comes through.
        return getattr(rep, "scalar", rep)

    # -- forward ----------------------------------------------------------------
    def forward(self, features=None, paths=None, return_nodes=False):
        if paths is None:
            raise ValueError(
                "TCPNetEncoder requires `paths=` (the per-sample file paths) to "
                "rebuild geometry from the PDB zip; the rank-dict is not used."
            )
        paths = list(paths)
        device = self.graph_proj[0].weight.device
        dim = self._scalar_dim
        batch, keep = self._build_batch(paths, device)

        # Scatter results back to the *input* order. This is essential: the
        # contrastive loss pairs z[:B] with z[B:], so output row i must always
        # correspond to paths[i]. Failed structures keep a zero row (their two
        # views fail identically, so the positive pair stays aligned, just
        # degenerate) rather than shifting every subsequent index.
        z = torch.zeros(len(paths), dim, device=device)
        if batch is None:
            if return_nodes:
                return z, torch.zeros(0, dim, device=device)
            return z

        out = self.encoder(batch)
        graph_emb = self._as_scalar(out["graph_embedding"])
        z_kept = self.graph_proj(graph_emb)
        keep_idx = torch.as_tensor(keep, device=device, dtype=torch.long)
        z = z.index_copy(0, keep_idx, z_kept)

        if return_nodes:
            node_emb = self._as_scalar(out["node_embedding"])
            return z, self.node_proj(node_emb)
        return z


# Back-compat alias: `--model topotein` historically built a class named
# `Topotein`. Keeping the name lets existing imports resolve to the new model.
Topotein = TCPNetEncoder


if __name__ == "__main__":
    # Standalone smoke test: build the model, featurise one real structure from
    # the zip, run a forward pass. Requires the ProteinWorkshop dep stack.
    _ensure_workshop_on_path()
    dev = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[smoke] device={dev}")
    model = TCPNetEncoder(scalar_dim=128).to(dev)
    print("[smoke] model built")

    # Pick any real sample path from the processed set.
    import glob
    sample = sorted(glob.glob(os.path.join("data", "hoan_processed", "*.pt")))[:4]
    paths = [os.path.basename(p) for p in sample]
    print(f"[smoke] featurising {len(paths)} structures: {paths}")
    z = model(paths=paths)
    print(f"[smoke] graph embedding: {tuple(z.shape)}  (expected (<=4, 128))")
    z2, h0 = model(paths=paths, return_nodes=True)
    print(f"[smoke] node embedding: {tuple(h0.shape)}")
    print("[smoke] OK")
