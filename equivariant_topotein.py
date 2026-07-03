"""
Equivariant Topotein Network Architecture
==========================================

Geometric, hierarchical message-passing counterpart of
``asymmetric_topotein.AsymmetricTopoNet``, with a *switchable* scalarisation that
spans two rungs of the geometry ablation (plan_experimentation_v2 §F, "invariant
scalaire vs équivariant"):

    invariant scalar      <      GVP norm-scalarisation      <      GCP frame-scalarisation
    (asymmetric_topotein)        (this file, scalarize='norm')      (this file, scalarize='frame')

Why the invariant model is "a weak version of topotein"
-------------------------------------------------------
``AsymmetricTopoNet`` keeps ``rank1['distance'] = ‖vector‖`` but discards
``rank1['vector']`` — the Cα→Cα displacement — so all orientation is gone before
layer 1. This module instead carries a **(scalar, vector)** tuple per cell and
updates them with geometric perceptrons over the same directed, hierarchical
topology (Rank0↔Rank1, Rank0→Rank2, {Rank0,Rank2}→Rank3).

Two scalarisations — and what each can and cannot see
-----------------------------------------------------
The decisive design axis is *how directional information enters the (invariant)
scalar channels* — the "scalarize" step:

* ``scalarize='norm'`` — **GVP** (Jing et al. 2021). A vector channel becomes its
  L2 norm ‖v‖: one scalar, magnitude only, direction discarded. The norm is
  invariant under all of O(3), **including reflection**, and the vector gate is a
  scalar — so the entire scalar pathway, and therefore the graph readout, is
  O(3)-invariant. **This cannot tell a structure from its mirror image** (it is
  *not* chirality-aware). Useful as a lightweight equivariant baseline rung.

* ``scalarize='frame'`` — **GCP** (Geometric Complete Perceptron, the primitive
  TCPNet/GCPNet is actually built on; see
  ``external/.../layers/gcp.py::scalarize``). A vector channel is **projected
  onto a complete local frame** ``F=(e1,e2,e3)`` → three *signed* scalars
  ``(v·e1, v·e2, v·e3)``, preserving direction. Because the frame is built with a
  **cross product** (``e3 = e1×e2``, a pseudovector that flips sign under
  reflection), the projection onto it is chirality-sensitive → the model is
  **SE(3)-equivariant and chirality-aware**. This recovers the geometric
  information GVP's norm bottleneck throws away — the actual motivation for going
  equivariant. (TCPNet's ``enable_e3_equivariance`` flag deliberately ``abs()``-es
  the cross-product component to *drop* chirality, confirming it is present by
  default.)

Relationship to the real Topotein / TCPNet
------------------------------------------
The real Topotein encoder is GCP/TCPNet on the full backbone (N, Cα, C, O) with
DSSP-derived cell complexes and dedicated per-edge frames. This file is a
**lightweight, Cα-only approximation**: the only raw geometry consumed is
``rank1['vector']`` (already collated and bucket-padded by the existing
pipeline), and the chiral local frames are built per-residue from three
consecutive Cα atoms (Cα_{i-1}, Cα_i, Cα_{i+1}) à la GCP, without re-lifting or
the ProteinWorkshop stack. It is intentionally *not* named ``Topotein`` to avoid
implying parity with the GCP/TCPNet encoder.

Equivariance guarantees (enforced below, verified in ``__main__``)
------------------------------------------------------------------
* Vector channels are equivariant (``V → V R``): bias-free channel mixing,
  scaling by invariant gates, aggregation with invariant attention weights.
* Local frames are equivariant (``F → F R``) and translation-invariant (built
  from coordinate differences only); the cross-product axis is a pseudovector,
  giving chirality sensitivity in ``frame`` mode.
* The graph readout is a scalar embedding (drop-in for the contrastive loss).
  In ``frame`` mode it is invariant under SE(3) and *changes* under reflection;
  in ``norm`` mode it is invariant under all of O(3).

Run ``python equivariant_topotein.py`` for the equivariance + chirality
self-test.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rbf import GaussianRBF


# ---------------------------------------------------------------------------
# Low-level geometry helpers (MPS-friendly: no torch.cross / einsum kernels)
# ---------------------------------------------------------------------------
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm along ``axis`` with a floor that avoids 0-gradient NaNs at x=0."""
    sq = torch.clamp(torch.sum(torch.square(x), dim=axis, keepdim=keepdims), min=eps)
    return torch.sqrt(sq) if sqrt else sq


def _normalize(x, axis=-1, eps=1e-8):
    return x / _norm_no_nan(x, axis=axis, keepdims=True, eps=eps)


def _cross(a, b):
    """Cross product along the last axis, written out (MPS lacks a robust
    torch.cross kernel on some builds)."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return torch.stack([ay * bz - az * by,
                        az * bx - ax * bz,
                        ax * by - ay * bx], dim=-1)


def build_node_frames(ca_coords, batch_idx):
    """Chiral local frame per residue from three consecutive Cα atoms.

    For residue i: e1 = chain tangent, e2 = in-plane normal (Gram-Schmidt),
    e3 = e1 × e2 (the pseudovector axis that carries chirality). Translation-
    invariant (uses only Cα differences) and rotation-equivariant
    (``F → F R``). Boundary residues (no sequence neighbour in the same protein)
    copy the adjacent interior frame. Returns ``(N, 3, 3)`` with frame axes as
    rows.
    """
    N = ca_coords.size(0)
    same_prev = batch_idx == torch.roll(batch_idx, 1, 0)
    same_next = batch_idx == torch.roll(batch_idx, -1, 0)
    same_prev = same_prev.clone(); same_prev[0] = False     # roll wrap-around
    same_next = same_next.clone(); same_next[-1] = False

    v_in = ca_coords - torch.roll(ca_coords, 1, 0)          # bond from i-1
    v_out = torch.roll(ca_coords, -1, 0) - ca_coords        # bond to i+1
    # Interior: v1 = outgoing bond, v2 = incoming bond (two distinct directions).
    # Boundary: fall back to the one available bond (frame fixed up below).
    v1 = torch.where(same_next.unsqueeze(-1), v_out, v_in)
    v2 = torch.where(same_prev.unsqueeze(-1), v_in, v_out)

    e1 = _normalize(v1)
    v2_orth = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
    e2 = _normalize(v2_orth)
    e3 = _cross(e1, e2)
    frames = torch.stack([e1, e2, e3], dim=-2)              # (N, 3, 3)

    # Boundary residues have a degenerate (collinear) frame -> copy the neighbour.
    is_first = ~same_prev
    is_last = ~same_next
    frames = torch.where(is_first.view(-1, 1, 1), torch.roll(frames, -1, 0), frames)
    frames = torch.where(is_last.view(-1, 1, 1), torch.roll(frames, 1, 0), frames)
    return frames


def _group_first_index(group, num_groups):
    """First (smallest) row index belonging to each group id. Used to give each
    SSE / protein a representative residue frame. Float scatter_reduce keeps it
    MPS-safe (no int64 amin kernel)."""
    N = group.size(0)
    ar = torch.arange(N, device=group.device, dtype=torch.float32)
    rep = torch.full((num_groups,), float(N), device=group.device)
    rep = rep.scatter_reduce(0, group, ar, reduce="amin", include_self=True)
    return rep.clamp(max=N - 1).long()


# ---------------------------------------------------------------------------
# Geometric perceptron (GVP norm-scalarisation OR GCP frame-scalarisation)
# ---------------------------------------------------------------------------
# A "scalar-vector" feature is a tuple ``(s, V)`` with ``s`` of shape
# ``(..., n_scalar)`` and ``V`` of shape ``(..., n_vector, 3)``.


class GCP(nn.Module):
    """Geometric perceptron ``(s, V) -> (s', V')``.

    ``scalarize='norm'`` reproduces a vector-gating GVP (direction discarded;
    O(3)-invariant scalars). ``scalarize='frame'`` is a GCP: vectors are
    projected onto a per-row local frame, yielding 3 *signed* scalars per hidden
    channel (direction + chirality; SE(3)-equivariant). The vector output and
    gating are identical in both modes — only how vectors feed the scalar
    pathway changes.
    """

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.silu, torch.sigmoid), vector_gate=True,
                 scalarize="frame"):
        super().__init__()
        assert scalarize in ("norm", "frame")
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        self.vector_gate = vector_gate
        self.scalarize = scalarize

        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)   # bias-free!
            n_scal = self.h_dim * (3 if scalarize == "frame" else 1)
            self.ws = nn.Linear(n_scal + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

    def forward(self, x, frames=None):
        if self.vi:
            s, v = x
            vh = self.wh(v.transpose(-1, -2))              # (..., 3, h)
            if self.scalarize == "frame":
                # Project each hidden vector channel onto the local frame axes:
                # signed scalars (v·e1, v·e2, v·e3). e3 is a pseudovector, so the
                # third scalar is chirality-sensitive.
                vh_hc = vh.transpose(-1, -2)               # (..., h, 3)
                proj = torch.matmul(vh_hc, frames.transpose(-1, -2))  # (..., h, 3)
                scal = proj.flatten(start_dim=-2)          # (..., 3h)  signed
            else:
                scal = _norm_no_nan(vh, axis=-2)           # (..., h)  magnitude
            s = self.ws(torch.cat([s, scal], dim=-1))
            if self.vo:
                vout = self.wv(vh).transpose(-1, -2)       # (..., vo, 3)
                if self.vector_gate:
                    gate = self.wsv(self.scalar_act(s) if self.scalar_act else s)
                    if self.vector_act:
                        gate = self.vector_act(gate)
                    vout = vout * gate.unsqueeze(-1)        # invariant gate
                elif self.vector_act:
                    vout = vout * self.vector_act(_norm_no_nan(vout, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                vout = torch.zeros(s.shape[:-1] + (self.vo, 3), device=s.device, dtype=s.dtype)

        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, vout) if self.vo else s


class GCPSequential(nn.Module):
    """Sequential container that threads the per-row ``frames`` through each GCP."""

    def __init__(self, *modules):
        super().__init__()
        self.gcps = nn.ModuleList(modules)

    def forward(self, x, frames=None):
        for m in self.gcps:
            x = m(x, frames)
        return x


class _AttnHead(nn.Module):
    """(s, V) -> invariant scalar logit. The GCP returns scalars only (built from
    scalar feats + frame projections / norms), so the attention weight is
    invariant and weighted vector messages stay equivariant."""

    def __init__(self, in_dims, s_dim, scalarize):
        super().__init__()
        self.gcp = GCP(in_dims, (s_dim, 0), scalarize=scalarize)
        self.lin = nn.Linear(s_dim, 1)

    def forward(self, x, frames):
        return self.lin(self.gcp(x, frames))


class GCPLayerNorm(nn.Module):
    """LayerNorm on scalars; RMS-norm (direction-preserving) on vectors."""

    def __init__(self, dims, eps=1e-8):
        super().__init__()
        self.s, self.v = dims
        self.eps = eps
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / (vn + self.eps)


class _VectorDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, v):
        if not self.training or self.p == 0:
            return v
        mask = torch.bernoulli((1 - self.p) * torch.ones(v.shape[:-1], device=v.device))
        return (mask / (1 - self.p)).unsqueeze(-1) * v


class GCPDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.sdrop = nn.Dropout(p)
        self.vdrop = _VectorDropout(p)

    def forward(self, x):
        if isinstance(x, tuple):
            return self.sdrop(x[0]), self.vdrop(x[1])
        return self.sdrop(x)


# ---------------------------------------------------------------------------
# Scalar-vector tuple helpers
# ---------------------------------------------------------------------------
def _sv_cat(*tuples):
    return torch.cat([t[0] for t in tuples], dim=-1), torch.cat([t[1] for t in tuples], dim=-2)


def _sv_index(x, idx):
    return x[0][idx], x[1][idx]


def _sv_scale(x, w):
    return x[0] * w, x[1] * w.unsqueeze(-1)


def _sv_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _scatter_add_sv(x, index, dim_size):
    s, v = x
    s_out = s.new_zeros((dim_size,) + s.shape[1:]).index_add_(0, index, s)
    v_out = v.new_zeros((dim_size,) + v.shape[1:]).index_add_(0, index, v)
    return s_out, v_out


def _scatter_mean_vec(vec, index, dim_size):
    summ = vec.new_zeros((dim_size,) + vec.shape[1:]).index_add_(0, index, vec)
    cnt = vec.new_zeros(dim_size, 1, 1).index_add_(0, index, vec.new_ones(vec.shape[0], 1, 1))
    return summ / cnt.clamp(min=1)


def _sv_mean(x, index, dim_size):
    s, v = x
    s_sum = s.new_zeros(dim_size, s.size(-1)).index_add_(0, index, s)
    v_sum = v.new_zeros((dim_size,) + v.shape[1:]).index_add_(0, index, v)
    cnt = s.new_zeros(dim_size, 1).index_add_(0, index, torch.ones_like(s[:, :1])).clamp(min=1)
    return s_sum / cnt, v_sum / cnt.unsqueeze(-1)


def _seg_softmax(logit, index, dim_size, dist_bias=None):
    if dist_bias is not None:
        logit = logit + dist_bias.unsqueeze(-1)
    e = torch.exp(torch.clamp(logit, min=-10.0, max=10.0))
    denom = e.new_zeros(dim_size, 1).index_add_(0, index, e)
    return e / (denom[index] + 1e-8)


# ---------------------------------------------------------------------------
# Equivariant hierarchical attention layer
# ---------------------------------------------------------------------------
class EquivariantTopoLayer(nn.Module):
    """One round of geometric, asymmetric, hierarchical message passing.

    Mirrors ``AsymmetricTopoAttentionLayer`` step-for-step. Each sub-step's
    perceptrons are fed the local frame of the cell-rank they operate on
    (residues -> node frames, contacts -> edge frames = source-node frames,
    SSEs/global -> a representative residue frame), matching GCP's convention of
    scalarising edge inputs in source-node frames.
    """

    def __init__(self, s_dim, v_dim, dropout=0.1, edge_attn_softmax=True,
                 dist_bias_gamma=0.0, scalarize="frame"):
        super().__init__()
        self.s_dim, self.v_dim = s_dim, v_dim
        self.edge_attn_softmax = edge_attn_softmax
        self.dist_bias_gamma = float(dist_bias_gamma)
        d = (s_dim, v_dim)

        def block(in_dims, out_dims):
            return GCPSequential(
                GCP(in_dims, out_dims, scalarize=scalarize),
                GCP(out_dims, out_dims, activations=(None, None), scalarize=scalarize),
            )

        cat_dims = (2 * s_dim, 2 * v_dim)

        self.edge_msg = block(cat_dims, d)
        self.edge_attn = _AttnHead(cat_dims, s_dim, scalarize)
        self.edge_ffn = block(d, d)
        self.norm_edge = GCPLayerNorm(d)

        self.sse_msg = block(cat_dims, d)
        self.sse_attn = _AttnHead(cat_dims, s_dim, scalarize)
        self.sse_ffn = block(d, d)
        self.norm_sse = GCPLayerNorm(d)

        self.node_msg = block(cat_dims, d)
        self.node_attn = _AttnHead(cat_dims, s_dim, scalarize)
        self.node_ffn = block(d, d)
        self.norm_node = GCPLayerNorm(d)

        self.global_gcp = block((3 * s_dim, 3 * v_dim), d)
        self.norm_global = GCPLayerNorm(d)

        self.drop = GCPDropout(dropout)

    def forward(self, h0, h1, h2, h3, src, dst, sse_map_0,
                batch_idx_0, batch_idx_2, B, F0, F1, F2, F3, edge_dist=None):
        N = h0[0].size(0)
        S = h2[0].size(0)

        dist_bias = (-self.dist_bias_gamma * edge_dist) if (self.dist_bias_gamma > 0 and edge_dist is not None) else None

        # 1. edges (Rank1) <- source nodes (Rank0); rows are edges -> edge frame F1
        src_in = _sv_cat(_sv_index(h0, src), h1)
        msg = self.edge_msg(src_in, F1)
        logit = self.edge_attn(src_in, F1)
        if self.edge_attn_softmax:
            attn = _seg_softmax(logit, src, N, dist_bias)
        else:
            attn = torch.sigmoid(logit if dist_bias is None else logit + dist_bias.unsqueeze(-1))
        h1 = self.norm_edge(_sv_add(h1, self.drop(self.edge_ffn(_sv_scale(msg, attn), F1))))

        # 2. SSEs (Rank2) <- member nodes (Rank0); message rows are nodes -> F0,
        #    aggregated SSE rows -> F2
        if S > 0:
            sse_in = _sv_cat(_sv_index(h2, sse_map_0), h0)
            msg = self.sse_msg(sse_in, F0)
            attn = _seg_softmax(self.sse_attn(sse_in, F0), sse_map_0, S)
            agg = _scatter_add_sv(_sv_scale(msg, attn), sse_map_0, S)
            h2 = self.norm_sse(_sv_add(h2, self.drop(self.sse_ffn(agg, F2))))

        # 3. residues (Rank0) <- incoming edges (Rank1); message rows are edges
        #    -> F1, aggregated node rows -> F0
        dst_in = _sv_cat(_sv_index(h0, dst), h1)
        msg = self.node_msg(dst_in, F1)
        attn = _seg_softmax(self.node_attn(dst_in, F1), dst, N, dist_bias)
        agg = _scatter_add_sv(_sv_scale(msg, attn), dst, N)
        h0 = self.norm_node(_sv_add(h0, self.drop(self.node_ffn(agg, F0))))

        # 4. global (Rank3) <- pooled nodes + SSEs; rows are proteins -> F3
        h0_pool = _sv_mean(h0, batch_idx_0, B)
        h2_pool = _sv_mean(h2, batch_idx_2, B) if S > 0 else (torch.zeros_like(h0_pool[0]), torch.zeros_like(h0_pool[1]))
        glob_in = _sv_cat(h3, h0_pool, h2_pool)
        h3 = self.norm_global(_sv_add(h3, self.drop(self.global_gcp(glob_in, F3))))

        return h0, h1, h2, h3


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class EquivariantTopoNet(nn.Module):
    """Geometric Topotein. Drop-in for ``AsymmetricTopoNet``: same
    ``forward(features, return_nodes=False)`` contract and normalized
    ``(B, scalar_dim)`` graph output, with an equivariant interior driven by the
    Cα→Cα displacement vectors the invariant model discards.

    Parameters of note
    -------------------
    scalarize : 'frame' (GCP, SE(3) + chirality, default) or 'norm' (GVP, O(3),
        reflection-blind). Picks the geometry ablation rung.
    vector_dim : number of vector channels carried per cell (default 16).
    rbf_dim : >0 swaps the stored 16-d sinusoidal distance encoding for a
        Gaussian-RBF expansion computed in-forward (plan_implementation §1);
        0 keeps the sinusoidal encoding for an iso-baseline.
    """

    def __init__(self, scalar_dim=128, vector_dim=16, num_layers=4, dropout=0.1,
                 use_positional_encoding=True, use_residue_features=True,
                 use_3di_features=True, edge_attn_softmax=True,
                 dist_bias_gamma=0.0, scalarize="frame",
                 rbf_dim=0, rbf_dmin=3.5, rbf_dmax=20.0):
        super().__init__()
        assert scalarize in ("norm", "frame")
        self.scalarize = scalarize
        self.use_positional_encoding = use_positional_encoding
        self.use_residue_features = use_residue_features
        self.use_3di_features = use_3di_features
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        self.rbf_dim = int(rbf_dim)
        if self.rbf_dim:
            self.rbf = GaussianRBF(num_rbf=self.rbf_dim, d_min=rbf_dmin, d_max=rbf_dmax)
        edge_scalar_in = 1 + (self.rbf_dim if self.rbf_dim else 16)

        d = (scalar_dim, vector_dim)
        emb = lambda in_dims: GCPSequential(
            GCP(in_dims, d, scalarize=scalarize),
            GCP(d, d, activations=(None, None), scalarize=scalarize),
        )
        self.rank0_emb = emb((64, 1))
        self.rank1_emb = emb((edge_scalar_in, 1))
        self.rank2_input_norm = nn.LayerNorm(12)
        self.rank2_emb = emb((12, 1))
        self.rank3_input_norm = nn.LayerNorm(10)
        self.rank3_emb = emb((10, 1))

        self.layers = nn.ModuleList([
            EquivariantTopoLayer(scalar_dim, vector_dim, dropout,
                                 edge_attn_softmax=edge_attn_softmax,
                                 dist_bias_gamma=dist_bias_gamma, scalarize=scalarize)
            for _ in range(num_layers)
        ])

        self.output_head = GCPSequential(
            GCP((2 * scalar_dim, 2 * vector_dim), d, scalarize=scalarize),
            GCP(d, (scalar_dim, 0), activations=(None, None), scalarize=scalarize),
        )
        self.output_norm = nn.LayerNorm(scalar_dim)

    def forward(self, features, return_nodes=False, return_vectors=False, return_repr=False):
        r0, r1, r2_feat, r3 = (features['rank0'], features['rank1'],
                               features['rank2_features'], features['rank3'])
        device = r0['aa'].device
        N = r0['aa'].shape[0]

        batch_idx_0 = features.get('batch_idx_0', torch.zeros(N, dtype=torch.long, device=device))
        batch_idx_2 = features.get('batch_idx_2', torch.zeros(r2_feat.size(0), dtype=torch.long, device=device))
        sse_map_0 = features['sse_map_0']
        B = r3['protein_size'].size(0) if 'protein_size' in r3 else 1

        src = r1['source'].flatten()
        dst = r1['target'].flatten()

        # --- Geometric input: the Cα→Cα edge displacement vectors (E,1,3) ----
        edge_vec = r1['vector'].flatten(0, 1).unsqueeze(1)
        node_seed_v = _scatter_mean_vec(edge_vec, src, N)
        S = r2_feat.size(0)
        sse_seed_v = (_scatter_mean_vec(node_seed_v, sse_map_0, S) if S > 0
                      else torch.zeros(0, 1, 3, device=device))
        glob_seed_v = _scatter_mean_vec(node_seed_v, batch_idx_0, B)

        # --- Local frames (only needed in 'frame' / GCP mode) ----------------
        if self.scalarize == "frame":
            F0 = build_node_frames(r0['ca_coords'].to(device), batch_idx_0)   # (N,3,3)
            F1 = F0[src]                                                       # edge = source-node frame
            F2 = F0[_group_first_index(sse_map_0, S)] if S > 0 else torch.zeros(0, 3, 3, device=device)
            F3 = F0[_group_first_index(batch_idx_0, B)]
        else:
            F0 = F1 = F2 = F3 = None

        # --- Scalar inputs (unchanged from the invariant model) --------------
        aa = r0['aa'] if self.use_residue_features else torch.zeros_like(r0['aa'])
        di = r0['3di'] if self.use_3di_features else torch.zeros_like(r0['3di'])
        pe = (r0['positional_encoding'] if self.use_positional_encoding
              else torch.zeros_like(r0['positional_encoding']))
        s0 = torch.cat([aa, di, r0['phi_psi'], pe], dim=-1)

        dist = r1['distance'].flatten()
        edge_enc = self.rbf(dist) if self.rbf_dim else r1['distance_encoding'].flatten(0, 1)
        s1 = torch.cat([dist.unsqueeze(-1), edge_enc], dim=-1)

        s2 = self.rank2_input_norm(r2_feat) if S > 0 else torch.empty(0, 12, device=device)

        p_size = r3['protein_size'].view(B, 1).to(device)
        rog = r3['radius_of_gyration'].view(B, 1).to(device)
        desc = r3['global_shape_descriptors'].view(B, 5).to(device)
        eigen = r3['global_eigenvalues'].view(B, 3).to(device)
        s3 = self.rank3_input_norm(torch.cat([p_size, rog, desc, eigen], dim=-1))

        # --- Embed each rank to a (scalar, vector) tuple ---------------------
        h0 = self.rank0_emb((s0, node_seed_v), F0)
        h1 = self.rank1_emb((s1, edge_vec), F1)
        if S > 0:
            h2 = self.rank2_emb((s2, sse_seed_v), F2)
        else:
            h2 = (torch.empty(0, self.scalar_dim, device=device),
                  torch.empty(0, self.vector_dim, 3, device=device))
        h3 = self.rank3_emb((s3, glob_seed_v), F3)

        for layer in self.layers:
            h0, h1, h2, h3 = layer(h0, h1, h2, h3, src, dst, sse_map_0,
                                   batch_idx_0, batch_idx_2, B, F0, F1, F2, F3, edge_dist=dist)

        # --- Equivariant pool + invariant readout ----------------------------
        h0_pool = _sv_mean(h0, batch_idx_0, B)
        # `repr` is the (invariant) readout representation h, pre-L2-normalization.
        # Eval uses normalize(h); the contrastive projection head consumes h.
        repr_ = self.output_norm(self.output_head(_sv_cat(h0_pool, h3), F3))
        out = F.normalize(repr_, p=2, dim=-1)
        out_formatted = out if B > 1 else out.squeeze(0)

        if return_repr:
            return repr_ if B > 1 else repr_.squeeze(0)
        if return_vectors:
            return out_formatted, h0_pool[1]                       # for the self-test
        if return_nodes:
            return out_formatted, h0[0]                            # invariant node scalars
        return out_formatted


# ---------------------------------------------------------------------------
# Equivariance + chirality self-test
# ---------------------------------------------------------------------------
def _random_features(N=40, K=16, S=5, B=2, device="cpu"):
    g = torch.Generator().manual_seed(0)
    per = N // B
    batch_idx_0 = torch.arange(B, device=device).repeat_interleave(per)
    if batch_idx_0.numel() < N:
        batch_idx_0 = torch.cat([batch_idx_0, torch.full((N - batch_idx_0.numel(),), B - 1, device=device)])

    def feats(coords):
        src_l, tgt_l = [], []
        for b in range(B):
            idx = (batch_idx_0 == b).nonzero(as_tuple=True)[0]
            dmat = torch.cdist(coords[idx], coords[idx])
            kk = min(K + 1, idx.numel())
            nn_idx = dmat.topk(kk, largest=False).indices[:, 1:]
            for r in range(idx.numel()):
                for c in idx[nn_idx[r]]:
                    src_l.append(int(idx[r])); tgt_l.append(int(c))
        src = torch.tensor(src_l, device=device); tgt = torch.tensor(tgt_l, device=device)
        Kp = src.numel() // N
        nu = N * Kp
        vec = coords[tgt] - coords[src]
        dist = vec.norm(dim=-1)
        return {
            'rank0': {'aa': torch.randn(N, 23, generator=g), '3di': torch.randn(N, 21, generator=g),
                      'phi_psi': torch.randn(N, 4, generator=g),
                      'positional_encoding': torch.randn(N, 16, generator=g), 'ca_coords': coords},
            'rank1': {'source': src[:nu].view(N, Kp), 'target': tgt[:nu].view(N, Kp),
                      'distance': dist[:nu].view(N, Kp),
                      'distance_encoding': torch.randn(N, Kp, 16, generator=g),
                      'vector': vec[:nu].view(N, Kp, 3)},
            'rank2_features': torch.randn(S, 12, generator=g),
            'rank3': {'protein_size': torch.full((B,), float(per)),
                      'radius_of_gyration': torch.rand(B, generator=g),
                      'global_shape_descriptors': torch.randn(B, 5, generator=g),
                      'global_eigenvalues': torch.rand(B, 3, generator=g)},
            'batch_idx_0': batch_idx_0,
            'batch_idx_2': (torch.arange(S, device=device) * B // S),
            'sse_map_0': torch.randint(0, S, (N,), generator=g),
        }

    return feats


def _transform(feats, M):
    import copy
    f = copy.deepcopy(feats)
    f['rank0']['ca_coords'] = feats['rank0']['ca_coords'] @ M.T
    f['rank1']['vector'] = feats['rank1']['vector'] @ M.T
    return f


if __name__ == "__main__":
    torch.manual_seed(0)
    build = _random_features()
    coords = torch.randn(40, 3)
    feats = build(coords)

    # Proper rotation (det +1) and a reflection (det -1).
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    R = Q
    Ref = torch.diag(torch.tensor([1.0, 1.0, -1.0]))

    for mode in ("norm", "frame"):
        model = EquivariantTopoNet(scalar_dim=64, vector_dim=8, num_layers=3, scalarize=mode).eval()
        with torch.no_grad():
            z, vpool = model(feats, return_vectors=True)
            z_rot, vpool_rot = model(_transform(feats, R), return_vectors=True)
            z_ref = model(_transform(feats, Ref))

        inv_err = (z - z_rot).abs().max().item()
        equi_err = (vpool_rot - vpool @ R.T).abs().max().item()
        used = (vpool_rot - vpool).abs().max().item()
        chir = (z - z_ref).abs().max().item()

        print(f"\n=== scalarize='{mode}' ===")
        print(f"  rotation: readout invariance err  : {inv_err:.2e}   (want ~0)")
        print(f"  rotation: vectors equivariance err: {equi_err:.2e}   (want ~0)")
        print(f"  geometry actually used (rot!=id)  : {used:.2e}   (want >0)")
        print(f"  REFLECTION: readout change        : {chir:.2e}")
        if mode == "norm":
            print("    -> expected ~0: GVP/norm is O(3)-invariant, chirality-BLIND")
            ok = inv_err < 1e-4 and equi_err < 1e-4 and used > 1e-3 and chir < 1e-4
        else:
            print("    -> expected LARGE: GCP/frame is SE(3) + chirality-AWARE")
            ok = inv_err < 1e-4 and equi_err < 1e-4 and used > 1e-3 and chir > 1e-2
        print("  ", "PASS" if ok else "FAIL")
