"""
Topotein: the full TCPNet encoder on the Protein Combinatorial Complex (PCC).
==============================================================================

This is a faithful, from-scratch implementation of the Topotein encoder
(Wang, Jamasb et al., *Topotein: Topological Deep Learning for Protein
Representation Learning*, arXiv:2509.03885) **adapted to the DeltaFold
experimentation protocol** (``documents/deltafold_protocol.tex``,
§Architecture). It replaces the deleted external ``tcpnet_adapter`` (which
vendored the ZW471 ProteinWorkshop stack and re-lifted from PDB on the fly) with
a self-contained module that consumes the *same batched rank-dict* the rest of
the pipeline already produces (see ``train.custom_collate`` /
``contrastive_collate``).

What makes this the *full* Topotein and not the lightweight
``equivariant_topotein.EquivariantTopoNet``
------------------------------------------------------------------------------
``EquivariantTopoNet`` is explicitly a Cα-only approximation that (a) uses
per-residue frames from three consecutive Cα atoms and (b) has **no inter-SSE
channel** — its rank-2 cells only ever gather their own member residues, so
secondary structures cannot talk to each other. Topotein's whole point is the
topology that lets them: this module implements

  * **edge-centric frames** ``F_(i,j)`` built per directed contact from the Cα
    positions of its endpoints (protocol §Edge-centric frames), with the
    cross-product axis giving chirality sensitivity;
  * the **TCP module** (protocol §TCP) — the rank-agnostic generalisation of the
    Geometry-Complete Perceptron: two bias-free vector MLPs ``V_s`` (→3 channels,
    scalarised on the rank frame → 9 signed scalars) and ``V_d`` (→ ``d_v/λ``
    bottleneck, norm-scalarised), a scalar output MLP and a sigmoid-gated vector
    output that keeps vectors SO(3)-equivariant while scalars stay invariant;
  * **outer-edge neighborhoods** ``N^{2→1}`` (protocol §Neighborhood functions of
    the PCC): the inter-SSE contacts (edges whose two endpoints lie in *different*
    SSEs) that carry tertiary packing between secondary structures — the channel
    the approximation lacks;
  * the **four-step hierarchical message passing** (protocol §Hierarchical
    message passing) that couples rank 1 and rank 2 directly: SSE features enter
    edge messages, and edge messages update SSEs through the outer-edge channel;
  * a **configurable message-passing order** via the ``tensor_diagram`` argument
    (protocol §Modular ordering / ``--tensor-diagram``), so the step order and the
    inter-SSE / global channels become experimental variables (the sweep of
    protocol §Design variables) without touching the model code;
  * a **dual readout** (protocol §Readout): node pooling (default) or the protein
    cell.

Equivariance guarantees (checked in ``__main__``)
-------------------------------------------------
Every attention weight is an invariant scalar and every learnable map is a TCP,
so scalar cochains stay SE(3)-invariant and vector cochains SO(3)-equivariant
through the whole stack. The scalar readout is therefore invariant to rigid
motion, permutation-invariant after pooling, and (in the default edge-frame mode)
inverts under reflection. Run ``python topotein.py`` for the self-test.

Drop-in contract
----------------
``forward(features, return_nodes=False, return_repr=False, return_vectors=False)``
takes the collated rank-dict and returns an ``ℓ2``-normalised ``(B, scalar_dim)``
graph embedding — the same contract as ``AsymmetricTopoNet`` /
``EquivariantTopoNet`` — so it slots into the contrastive loop, MoCo wrapper and
``extract_embeddings`` unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rbf import GaussianRBF
from equivariant_topotein import (
    _cross, _normalize, _norm_no_nan, _seg_softmax,
    _scatter_mean_vec, _sv_cat, _sv_index, _sv_scale, _sv_add,
    _scatter_add_sv, _sv_mean, GCPLayerNorm, GCPDropout, _group_first_index,
)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def _scatter_mean_frames(frames, index, dim_size):
    """Mean of per-row (3, 3) frames within each group. Averaging equivariant
    frames stays equivariant (F → F R), so scalarising a vector on the mean frame
    keeps the resulting scalars invariant — this is exactly the per-rank
    scalarisation ``S^{(r)}_i`` of the protocol (average the projections over a
    cell's incident/outer edges)."""
    out = frames.new_zeros((dim_size, 3, 3)).index_add_(0, index, frames)
    cnt = frames.new_zeros(dim_size, 1, 1).index_add_(
        0, index, frames.new_ones(frames.shape[0], 1, 1))
    return out / cnt.clamp(min=1)


def build_edge_frames(ca_centered, src, dst, eps=1e-8):
    """Edge-centric frame ``F_(i,j) = (e1, e2, e3)`` for every directed contact
    (protocol §Edge-centric frames):

        e1 = (x_j - x_i) / ‖·‖          (bond direction)
        e2 = (x_j × x_i) / ‖·‖          (pseudovector axis -> chirality)
        e3 = e1 × e2

    Built from centred Cα coordinates so it is invariant to global translation and
    equivariant to rotation (``F → F R``); the cross-product axis flips under
    reflection, making the projection chirality-sensitive. Returns ``(E, 3, 3)``
    with frame axes as rows. Degenerate edges (``x_j ∥ x_i``) fall back to a stable
    complement so the frame stays orthonormal-ish."""
    xi, xj = ca_centered[src], ca_centered[dst]
    e1 = _normalize(xj - xi, eps=eps)
    cr = _cross(xj, xi)
    cr_norm = _norm_no_nan(cr, keepdims=True, eps=eps)
    # Fallback axis when x_j ∥ x_i (origin-collinear): any vector not parallel to e1.
    fallback = _cross(e1, torch.roll(e1, 1, dims=-1))
    e2 = torch.where(cr_norm > 1e-4, cr / cr_norm, _normalize(fallback, eps=eps))
    e3 = _cross(e1, e2)
    return torch.stack([e1, e2, e3], dim=-2)


def build_protein_frames(ca_centered, batch_idx, B, eps=1e-6):
    """Protein (rank-3) frame from the three principal eigenvectors of the Cα point
    cloud (protocol §Edge-centric frames, last paragraph), sign-disambiguated by
    the SVD sign convention of Bro et al. (2008): flip each eigenvector so that
    ``Σ_i sign(p_i)·p_i²`` is positive, where ``p_i`` is the projection of residue
    ``i`` on that axis. Returns ``(B, 3, 3)`` with eigenvectors (λ1≥λ2≥λ3) as rows.

    Eigen-decomposition runs on CPU: ``torch.linalg.eigh`` is unsupported on the
    MPS backend, and a per-protein 3×3 solve is negligible."""
    # Per-protein covariance via scatter (coords are already centred).
    outer = ca_centered.unsqueeze(-1) * ca_centered.unsqueeze(-2)          # (N,3,3)
    cov = outer.new_zeros(B, 3, 3).index_add_(0, batch_idx, outer)
    cnt = outer.new_zeros(B, 1, 1).index_add_(
        0, batch_idx, outer.new_ones(outer.shape[0], 1, 1)).clamp(min=1)
    cov = cov / cnt
    cov = cov + eps * torch.eye(3, device=cov.device).unsqueeze(0)

    evals, evecs = torch.linalg.eigh(cov.cpu())                            # ascending
    evecs = evecs.to(cov.device)
    # Reorder to λ1≥λ2≥λ3 and put eigenvectors in columns order [::-1].
    evecs = evecs.flip(-1)                                                 # (B,3,3) cols

    # Bro et al. sign disambiguation, vectorised.
    proj = torch.matmul(ca_centered.unsqueeze(-2), evecs[batch_idx]).squeeze(-2)  # (N,3)
    signed = torch.sign(proj) * proj * proj
    s = signed.new_zeros(B, 3).index_add_(0, batch_idx, signed)
    sign = torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))    # (B,3)
    evecs = evecs * sign.unsqueeze(-2)                                     # flip columns
    return evecs.transpose(-1, -2)                                         # axes as rows


# ---------------------------------------------------------------------------
# The Topology-Complete Perceptron (TCP)
# ---------------------------------------------------------------------------
class TCP(nn.Module):
    """Rank-agnostic geometric perceptron ``(s, V) -> (s', V')`` (protocol §TCP).

    ``h_v ∈ R^{d_v×3}`` is reduced two ways by *bias-free* channel mixers:

        s_vec = V_s(h_v) ∈ R^{3×3}     (3 vector channels, scalarised on the frame)
        z     = V_d(h_v) ∈ R^{(d_v/λ)×3}   (bottleneck, magnitude-scalarised)

    The scalar stream ``h'_s = (h_s, S^{(r)}(s_vec), ‖z‖)`` is mapped by ``S_out``;
    the vector output ``h'_v = V_u(z)`` is gated by ``σ_g(S_gate(h_{s,out}))`` — a
    scalar gate that modulates vector *magnitude* while preserving equivariance.

    ``scalarize='frame'`` projects ``s_vec`` onto the per-row frame (9 signed
    scalars, chirality-aware); ``scalarize='norm'`` takes ‖s_vec‖ (3 scalars,
    O(3)-invariant, reflection-blind) — the geometry ablation rung of the
    protocol sweep (edge-centric vs. norm frames). No pointwise nonlinearity is
    applied to vector channels (that would break equivariance)."""

    def __init__(self, in_dims, out_dims, bottleneck=3, vector_gate=True,
                 activations=(F.silu, torch.sigmoid), scalarize="frame"):
        super().__init__()
        assert scalarize in ("norm", "frame")
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalarize = scalarize
        self.scalar_act, self.vector_act = activations
        self.vector_gate = vector_gate

        if self.vi:
            self.Vs = nn.Linear(self.vi, 3, bias=False)                 # -> 3 vec channels
            zc = max(1, (self.vo if self.vo else self.vi) // bottleneck)
            self.zc = zc
            self.Vd = nn.Linear(self.vi, zc, bias=False)               # -> bottleneck
            n_proj = 9 if scalarize == "frame" else 3
            self.Sout = nn.Linear(self.si + n_proj + zc, self.so)
            if self.vo:
                self.Vu = nn.Linear(zc, self.vo, bias=False)          # -> vector out
                if vector_gate:
                    self.Sgate = nn.Linear(self.so, self.vo)
        else:
            self.Sout = nn.Linear(self.si, self.so)

    def forward(self, x, frames=None):
        if self.vi:
            s, v = x                                                   # v: (...,vi,3)
            vt = v.transpose(-1, -2)                                   # (...,3,vi)
            s_vec = self.Vs(vt).transpose(-1, -2)                      # (...,3,3)
            z = self.Vd(vt).transpose(-1, -2)                         # (...,zc,3)
            if self.scalarize == "frame":
                proj = torch.matmul(s_vec, frames.transpose(-1, -2))  # (...,3,3) signed
                scal = proj.flatten(start_dim=-2)                     # (...,9)
            else:
                scal = _norm_no_nan(s_vec, axis=-1)                   # (...,3) magnitude
            znorm = _norm_no_nan(z, axis=-1)                          # (...,zc)
            sout = self.Sout(torch.cat([s, scal, znorm], dim=-1))
            if self.scalar_act:
                sout = self.scalar_act(sout)
            if self.vo:
                vout = self.Vu(z.transpose(-1, -2)).transpose(-1, -2)  # (...,vo,3)
                if self.vector_gate:
                    gate = self.Sgate(sout)
                    if self.vector_act:
                        gate = self.vector_act(gate)
                    vout = vout * gate.unsqueeze(-1)                   # invariant gate
                return sout, vout
            return sout
        # scalar-only input
        s = self.Sout(x[0] if isinstance(x, tuple) else x)
        if self.scalar_act:
            s = self.scalar_act(s)
        if self.vo:
            vout = torch.zeros(s.shape[:-1] + (self.vo, 3), device=s.device, dtype=s.dtype)
            return s, vout
        return s


class TCPSequential(nn.Module):
    """Threads the per-row ``frames`` through a stack of TCP modules."""

    def __init__(self, *modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x, frames=None):
        for m in self.mods:
            x = m(x, frames)
        return x


class _AttnHead(nn.Module):
    """(s, V) -> invariant scalar logit. Built from a scalar-out TCP, so the
    attention weight is SE(3)-invariant and the weighted vector messages stay
    equivariant."""

    def __init__(self, in_dims, s_dim, scalarize):
        super().__init__()
        self.tcp = TCP(in_dims, (s_dim, 0), scalarize=scalarize)
        self.lin = nn.Linear(s_dim, 1)

    def forward(self, x, frames):
        return self.lin(self.tcp(x, frames))


def _block(in_dims, out_dims, scalarize):
    """Two-TCP residual block (message / feed-forward function φ)."""
    return TCPSequential(
        TCP(in_dims, out_dims, scalarize=scalarize),
        TCP(out_dims, out_dims, activations=(None, None), scalarize=scalarize),
    )


def _zeros_like_sv(ref_s, ref_v, rows):
    return (ref_s.new_zeros(rows, ref_s.size(-1)),
            ref_v.new_zeros((rows,) + ref_v.shape[1:]))


# ---------------------------------------------------------------------------
# One interaction layer: Topotein's four-step hierarchical message passing
# ---------------------------------------------------------------------------
class TopoteinLayer(nn.Module):
    """One interaction layer (protocol §Hierarchical message passing).

    The four steps — (1) edge messages with SSE context, (2) SSE update through
    inner + outer edges, (3) residue refinement, (4) protein update — each are a
    cochain push-forward along a neighborhood, and a layer is their ordered
    composition. The order and which channels are active are controlled by the
    parent network's ``tensor_diagram``; the module always builds the full
    parameter set (fixed shape → checkpoint-stable) and *zeros the inputs* of a
    disabled channel, exactly as the pipeline's shortcut-mitigation flags do."""

    def __init__(self, s_dim, v_dim, dropout=0.1, edge_attn_softmax=True,
                 scalarize="frame"):
        super().__init__()
        self.s_dim, self.v_dim = s_dim, v_dim
        self.edge_attn_softmax = edge_attn_softmax
        d = (s_dim, v_dim)

        # Step 1: edge <- {src res, dst res, edge, src SSE, dst SSE}  (5 cells)
        self.edge_msg = _block((5 * s_dim, 5 * v_dim), d, scalarize)
        self.edge_attn = _AttnHead((5 * s_dim, 5 * v_dim), s_dim, scalarize)
        self.edge_ffn = _block(d, d, scalarize)
        self.norm_edge = GCPLayerNorm(d)

        # Step 2: SSE <- {SSE, member residues, inner edges, outer-edge messages}
        self.sse_upd = _block((4 * s_dim, 4 * v_dim), d, scalarize)
        self.norm_sse = GCPLayerNorm(d)

        # Step 3: residue <- {residue, parent SSE, incoming edge messages}
        self.node_upd = _block((3 * s_dim, 3 * v_dim), d, scalarize)
        self.norm_node = GCPLayerNorm(d)

        # Step 4: protein <- {protein, pooled residues, pooled SSEs}
        self.global_upd = _block((3 * s_dim, 3 * v_dim), d, scalarize)
        self.norm_global = GCPLayerNorm(d)

        self.drop = GCPDropout(dropout)

    # -- individual steps ---------------------------------------------------
    def _step_edge(self, ctx):
        h0, h1, h2 = ctx['h0'], ctx['h1'], ctx['h2']
        src, dst, F1 = ctx['src'], ctx['dst'], ctx['F1']
        N = h0[0].size(0)
        # SSE context n_i for each residue (zeroed when the inter-SSE channel is off).
        if ctx['use_sse_ctx'] and h2[0].size(0) > 0:
            n = _sv_index(h2, ctx['sse_map_0'])
            n_src, n_dst = _sv_index(n, src), _sv_index(n, dst)
        else:
            n_src = n_dst = _zeros_like_sv(h1[0], h1[1], src.size(0))
        edge_in = _sv_cat(_sv_index(h0, src), _sv_index(h0, dst), h1, n_src, n_dst)
        msg = self.edge_msg(edge_in, F1)
        logit = self.edge_attn(edge_in, F1)
        if self.edge_attn_softmax:
            attn = _seg_softmax(logit, src, N)
        else:
            attn = torch.sigmoid(logit)
        h1 = self.norm_edge(_sv_add(h1, self.drop(self.edge_ffn(_sv_scale(msg, attn), F1))))
        ctx['h1'] = h1
        # Attention-gated edge message consumed by ranks 0/2 downstream.
        ctx['edge_message'] = (h1[0] * attn, h1[1])

    def _step_sse(self, ctx):
        h0, h2 = ctx['h0'], ctx['h2']
        S = h2[0].size(0)
        if S == 0:
            return
        sse_map_0, src, F2 = ctx['sse_map_0'], ctx['src'], ctx['F2']
        m = ctx['edge_message']
        # member residues (B^{2->0})
        agg_nodes = _sv_mean(h0, sse_map_0, S)
        if ctx['use_edge_to_sse']:
            inner, outer = ctx['inner_mask'], ~ctx['inner_mask']
            sse_src = sse_map_0[src]
            # inner edges (B^{2->1}): both endpoints in this SSE
            agg_inner = _sv_mean(_sv_index(ctx['h1'], inner.nonzero(as_tuple=True)[0]),
                                 sse_src[inner], S)
            # outer-edge messages (N^{2->1}): edges that START in this SSE, end elsewhere
            oidx = outer.nonzero(as_tuple=True)[0]
            agg_outer = _sv_mean(_sv_index(m, oidx), sse_src[outer], S)
        else:
            agg_inner = _zeros_like_sv(h2[0], h2[1], S)
            agg_outer = _zeros_like_sv(h2[0], h2[1], S)
        sse_in = _sv_cat(h2, agg_nodes, agg_inner, agg_outer)
        ctx['h2'] = self.norm_sse(_sv_add(h2, self.drop(self.sse_upd(sse_in, F2))))

    def _step_node(self, ctx):
        h0, h2 = ctx['h0'], ctx['h2']
        dst, F0 = ctx['dst'], ctx['F0']
        N = h0[0].size(0)
        # parent SSE (updated) per residue
        if ctx['use_sse_ctx'] and h2[0].size(0) > 0:
            m2 = _sv_index(h2, ctx['sse_map_0'])
        else:
            m2 = _zeros_like_sv(h0[0], h0[1], N)
        # incoming edge messages (transpose of B^{1->0})
        agg_edge = _sv_mean(ctx['edge_message'], dst, N)
        node_in = _sv_cat(h0, m2, agg_edge)
        ctx['h0'] = self.norm_node(_sv_add(h0, self.drop(self.node_upd(node_in, F0))))

    def _step_protein(self, ctx):
        h0, h2, h3 = ctx['h0'], ctx['h2'], ctx['h3']
        B, F3 = ctx['B'], ctx['F3']
        h0_pool = _sv_mean(h0, ctx['batch_idx_0'], B)
        if h2[0].size(0) > 0:
            h2_pool = _sv_mean(h2, ctx['batch_idx_2'], B)
        else:
            h2_pool = _zeros_like_sv(h3[0], h3[1], B)
        glob_in = _sv_cat(h3, h0_pool, h2_pool)
        ctx['h3'] = self.norm_global(_sv_add(h3, self.drop(self.global_upd(glob_in, F3))))

    _STEP_FN = {'edge': _step_edge, 'sse': _step_sse,
                'node': _step_node, 'protein': _step_protein}

    def forward(self, ctx):
        for name in ctx['order']:
            self._STEP_FN[name](self, ctx)
        return ctx


# ---------------------------------------------------------------------------
# Tensor diagrams (protocol §Modular ordering / --tensor-diagram)
# ---------------------------------------------------------------------------
# Each preset resolves to (step order, channel toggles). The full declarative DSL
# of the protocol is represented here by the named diagrams that the sweep table
# (§Design variables) actually asks for; a custom order can also be passed as a
# comma-separated string, e.g. "edge,node,sse,protein".
_DIAGRAMS = {
    # Topotein default: bottom-up-then-down with the inter-SSE outer-edge channel.
    'default':     dict(order=('edge', 'sse', 'node', 'protein'),
                        use_sse_ctx=True,  use_edge_to_sse=True,  use_rank3=True),
    # Residue-hub control: SSEs/edges communicate only through rank 0 (no N^{2->1},
    # no SSE context in edge messages) — isolates the effect of the inter-SSE channel.
    'residue_hub': dict(order=('edge', 'sse', 'node', 'protein'),
                        use_sse_ctx=False, use_edge_to_sse=False, use_rank3=True),
    # No-rank-3: drop the protein channel; readout must pool residues.
    'no_rank3':    dict(order=('edge', 'sse', 'node'),
                        use_sse_ctx=True,  use_edge_to_sse=True,  use_rank3=False),
    # Reordered sweep example: refine residues before SSEs.
    'reordered':   dict(order=('edge', 'node', 'sse', 'protein'),
                        use_sse_ctx=True,  use_edge_to_sse=True,  use_rank3=True),
}


def resolve_diagram(spec):
    if isinstance(spec, dict):
        return spec
    if spec in _DIAGRAMS:
        return dict(_DIAGRAMS[spec])
    # Custom order string -> full-channel diagram in that order.
    order = tuple(s.strip() for s in spec.split(',') if s.strip())
    valid = {'edge', 'sse', 'node', 'protein'}
    assert order and set(order) <= valid, f"bad tensor_diagram '{spec}'"
    return dict(order=order, use_sse_ctx=True, use_edge_to_sse=True,
                use_rank3=('protein' in order))


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class TCPNet(nn.Module):
    """The Topotein encoder (TCPNet) on the PCC. Drop-in for
    ``AsymmetricTopoNet`` / ``EquivariantTopoNet``: same forward contract and
    ``ℓ2``-normalised ``(B, scalar_dim)`` graph output.

    Parameters of note
    -------------------
    scalarize : 'frame' (edge-centric GCP/TCP frames, SE(3)+chirality; default) or
        'norm' (O(3), reflection-blind) — the geometry ablation rung.
    vector_dim : vector channels carried per cell (protocol d_v; default 16).
    bottleneck : TCP bottleneck λ (protocol; default 3).
    tensor_diagram : 'default' | 'residue_hub' | 'no_rank3' | 'reordered' | a
        custom "edge,node,sse,protein" order (protocol --tensor-diagram).
    readout : 'node' (pool residues; default) or 'protein' (use the rank-3 cell).
    rbf_dim : >0 swaps the stored 16-d sinusoidal edge-distance encoding for a
        Gaussian-RBF expansion (protocol §Featurization RBF); 0 keeps sinusoidal.
    """

    def __init__(self, scalar_dim=128, vector_dim=16, num_layers=4, dropout=0.1,
                 bottleneck=3, tensor_diagram='default', readout='node',
                 use_positional_encoding=True, use_residue_features=True,
                 use_3di_features=True, edge_attn_softmax=True, scalarize="frame",
                 rbf_dim=0, rbf_dmin=3.5, rbf_dmax=20.0):
        super().__init__()
        assert scalarize in ("norm", "frame")
        assert readout in ("node", "protein")
        self.scalarize = scalarize
        self.readout = readout
        self.diagram = resolve_diagram(tensor_diagram)
        if readout == 'protein' and not self.diagram['use_rank3']:
            raise ValueError("readout='protein' needs a diagram with the rank-3 channel")
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

        def emb(in_dims):
            return TCPSequential(
                TCP(in_dims, d, bottleneck=bottleneck, scalarize=scalarize),
                TCP(d, d, activations=(None, None), bottleneck=bottleneck, scalarize=scalarize),
            )

        # GVP-style input layer norm on the raw scalars before the embedding TCP.
        self.rank0_emb = emb((64, 1))
        self.rank1_emb = emb((edge_scalar_in, 1))
        self.rank2_input_norm = nn.LayerNorm(12)
        self.rank2_emb = emb((12, 1))
        self.rank3_input_norm = nn.LayerNorm(10)
        self.rank3_emb = emb((10, 1))

        self.layers = nn.ModuleList([
            TopoteinLayer(scalar_dim, vector_dim, dropout,
                          edge_attn_softmax=edge_attn_softmax, scalarize=scalarize)
            for _ in range(num_layers)
        ])

        # Readout: node pooling concatenates pooled residues with the protein cell
        # (when present); protein readout uses the rank-3 cell alone.
        head_vin = 2 if (readout == 'node' and self.diagram['use_rank3']) else 1
        self.output_head = TCPSequential(
            TCP((head_vin * scalar_dim, head_vin * vector_dim), d,
                bottleneck=bottleneck, scalarize=scalarize),
            TCP(d, (scalar_dim, 0), activations=(None, None),
                bottleneck=bottleneck, scalarize=scalarize),
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
        S = r2_feat.size(0)

        src = r1['source'].flatten()
        dst = r1['target'].flatten()

        # --- Geometry: Cα→Cα edge displacements seed every rank's vector channel ---
        edge_vec = r1['vector'].flatten(0, 1).unsqueeze(1)                 # (E,1,3)
        node_seed_v = _scatter_mean_vec(edge_vec, src, N)
        sse_seed_v = (_scatter_mean_vec(node_seed_v, sse_map_0, S) if S > 0
                      else torch.zeros(0, 1, 3, device=device))
        glob_seed_v = _scatter_mean_vec(node_seed_v, batch_idx_0, B)

        # --- Frames ----------------------------------------------------------
        if self.scalarize == "frame":
            ca = r0['ca_coords'].to(device)
            cbar = ca.new_zeros(B, 3).index_add_(0, batch_idx_0, ca)
            cnt = ca.new_zeros(B, 1).index_add_(0, batch_idx_0, ca.new_ones(N, 1)).clamp(min=1)
            ca_c = ca - (cbar / cnt)[batch_idx_0]                          # centred
            F1 = build_edge_frames(ca_c, src, dst)                        # (E,3,3) edge frames
            F0 = _scatter_mean_frames(F1, src, N)                        # residue = mean incident-edge frame
            if S > 0:
                sse_src = sse_map_0[src]
                outer = sse_src != sse_map_0[dst]
                if outer.any():
                    F2 = _scatter_mean_frames(F1[outer], sse_src[outer], S)  # SSE = mean outer-edge frame
                else:
                    F2 = _scatter_mean_frames(F1, sse_src, S)
            else:
                F2 = torch.zeros(0, 3, 3, device=device)
            F3 = build_protein_frames(ca_c, batch_idx_0, B)               # (B,3,3) principal axes
        else:
            F0 = F1 = F2 = F3 = None

        # --- Scalar inputs ---------------------------------------------------
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

        # --- Hierarchical message passing ------------------------------------
        inner_mask = (sse_map_0[src] == sse_map_0[dst]) if S > 0 else None
        ctx = dict(h0=h0, h1=h1, h2=h2, h3=h3, src=src, dst=dst,
                   sse_map_0=sse_map_0, batch_idx_0=batch_idx_0, batch_idx_2=batch_idx_2,
                   B=B, F0=F0, F1=F1, F2=F2, F3=F3, inner_mask=inner_mask,
                   order=self.diagram['order'], use_sse_ctx=self.diagram['use_sse_ctx'],
                   use_edge_to_sse=self.diagram['use_edge_to_sse'])
        for layer in self.layers:
            ctx = layer(ctx)
        h0, h2, h3 = ctx['h0'], ctx['h2'], ctx['h3']

        # --- Readout ---------------------------------------------------------
        if self.readout == 'protein':
            pooled = h3                                                    # rank-3 cell
        else:
            h0_pool = _sv_mean(h0, batch_idx_0, B)                        # node pooling
            pooled = _sv_cat(h0_pool, h3) if self.diagram['use_rank3'] else h0_pool
        repr_ = self.output_norm(self.output_head(pooled, F3))
        out = F.normalize(repr_, p=2, dim=-1)
        out_formatted = out if B > 1 else out.squeeze(0)

        if return_repr:
            return repr_ if B > 1 else repr_.squeeze(0)
        if return_vectors:
            return out_formatted, (h0_pool[1] if self.readout == 'node' else h3[1])
        if return_nodes:
            return out_formatted, h0[0]
        return out_formatted


# Backwards-compatible alias: the pipeline instantiates ``Topotein`` for --model topotein.
Topotein = TCPNet


# ---------------------------------------------------------------------------
# Equivariance + chirality self-test (mirrors equivariant_topotein.py)
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
        # SSE map: contiguous runs so within-protein SSEs are well defined.
        sse_map = (batch_idx_0 * S + (torch.arange(N) % S)).long()
        return {
            'rank0': {'aa': torch.randn(N, 23, generator=g), '3di': torch.randn(N, 21, generator=g),
                      'phi_psi': torch.randn(N, 4, generator=g),
                      'positional_encoding': torch.randn(N, 16, generator=g), 'ca_coords': coords},
            'rank1': {'source': src[:nu].view(N, Kp), 'target': tgt[:nu].view(N, Kp),
                      'distance': dist[:nu].view(N, Kp),
                      'distance_encoding': torch.randn(N, Kp, 16, generator=g),
                      'vector': vec[:nu].view(N, Kp, 3)},
            'rank2_features': torch.randn(B * S, 12, generator=g),
            'rank3': {'protein_size': torch.full((B,), float(per)),
                      'radius_of_gyration': torch.rand(B, generator=g),
                      'global_shape_descriptors': torch.randn(B, 5, generator=g),
                      'global_eigenvalues': torch.rand(B, 3, generator=g)},
            'batch_idx_0': batch_idx_0,
            'batch_idx_2': torch.arange(B * S, device=device) // S,
            'sse_map_0': sse_map,
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

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    R = Q
    Ref = torch.diag(torch.tensor([1.0, 1.0, -1.0]))

    for mode in ("norm", "frame"):
        for diagram in ("default", "residue_hub", "no_rank3"):
            model = TCPNet(scalar_dim=64, vector_dim=8, num_layers=3,
                           scalarize=mode, tensor_diagram=diagram).eval()
            with torch.no_grad():
                z, vpool = model(feats, return_vectors=True)
                z_rot, vpool_rot = model(_transform(feats, R), return_vectors=True)
                z_ref = model(_transform(feats, Ref))
            inv_err = (z - z_rot).abs().max().item()
            equi_err = (vpool_rot - vpool @ R.T).abs().max().item()
            used = (vpool_rot - vpool).abs().max().item()
            chir = (z - z_ref).abs().max().item()
            print(f"\n=== scalarize='{mode}' diagram='{diagram}' ===")
            print(f"  rotation invariance err : {inv_err:.2e}  (want ~0)")
            print(f"  vector equivariance err : {equi_err:.2e}  (want ~0)")
            print(f"  geometry used (rot!=id) : {used:.2e}  (want >0)")
            print(f"  REFLECTION readout change: {chir:.2e}")
            if mode == "norm":
                # GVP/norm scalarisation is O(3)-invariant -> reflection changes
                # nothing (chirality-blind): the readout is bit-for-bit identical.
                ok = inv_err < 1e-4 and equi_err < 1e-4 and used > 1e-3 and chir < 1e-6
            else:
                # Edge frames are chiral (cross-product axis) -> the readout DOES
                # respond to reflection. Chirality enters only through the edge/
                # SSE frames during message passing (the protein eigen-frame is a
                # proper frame), so at random init the signal is modest but
                # strictly nonzero; it grows once the vector channels carry real
                # (non-cancelling) directional features. The decisive contrast is
                # frame (nonzero) vs norm (exactly zero above).
                ok = inv_err < 1e-4 and equi_err < 1e-4 and used > 1e-3 and chir > 1e-4
            print("  ", "PASS" if ok else "FAIL")
