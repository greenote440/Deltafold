"""
Asymmetric Topotein Network Architecture
Implementation of the HOAN-symmetry-aware topological attention network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricTopoAttentionLayer(nn.Module):
    """
    Asymmetric Topo-Attention Layer
    Implements directed, asymmetric message passing using Attention
    to preserve HOAN symmetry.
    """
    def __init__(self, dim, dropout=0.1, edge_attn_softmax=False,
                 dist_bias_gamma=0.0, detach_h3=False):
        super().__init__()
        self.dim = dim

        # TM-score analysis fixes (tm_score_analysis.md §5.3/5.4/5.7). All three are
        # forward-behavior-only and add NO parameters, so checkpoints stay loadable.
        # Defaults reproduce the ORIGINAL behavior so old checkpoints (whose saved
        # model_config lacks these keys) run faithfully; the training entrypoint
        # turns the recommended behavior on and records it in model_config.
        #   edge_attn_softmax=True -> normalise the Rank0->Rank1 edge attention over
        #       each source node's K edges (§5.3) instead of an element-wise sigmoid
        #       gate that saturates to unweighted mean aggregation.
        #   dist_bias_gamma>0      -> add a -gamma*distance geometric bias to edge and
        #       node attention logits (§5.4), anchoring attention to 3D-proximal
        #       neighbors rather than feature/sequence similarity.
        #   detach_h3=True         -> detach pooled features feeding the per-layer h3
        #       update (§5.7) to break the global shortcut gradient path.
        self.edge_attn_softmax = edge_attn_softmax
        self.dist_bias_gamma = float(dist_bias_gamma)
        self.detach_h3 = detach_h3

        # Diagnostics hook (inert during training). When `capture_attn` is set
        # True from outside, the layer stashes the last node-attention weights and
        # their (src, dst) indices so diagnostics.py can test whether attention is
        # driven by sequence position rather than 3D structural context (report 3.2).
        self.capture_attn = False
        self.last_node_attn = None
        self.last_attn_src = None
        self.last_attn_dst = None

        # Rank 0 -> Rank 1 (Edge gathers from Source)
        self.W_q_edge = nn.Linear(dim, dim)
        self.W_kv_src = nn.Linear(dim, dim * 2)
        
        # Rank 1 -> Rank 0 (Target updates from Edge)
        self.W_q_node = nn.Linear(dim, dim)
        self.W_kv_dst = nn.Linear(dim, dim * 2)

        # Rank 0 -> Rank 2 (SSE gathers from Nodes)
        self.W_q_sse = nn.Linear(dim, dim)
        self.W_kv_node_sse = nn.Linear(dim, dim * 2)

        # Feed Forward Networks
        self.edge_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        self.sse_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self.node_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self.norm_edge = nn.LayerNorm(dim)
        self.norm_sse = nn.LayerNorm(dim)
        self.norm_node = nn.LayerNorm(dim)
        self.norm_global = nn.LayerNorm(dim)
        
        # Global Pooling Update
        self.W_global = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, h0, h1, h2, h3, src, dst, sse_map_0=None, batch_idx_0=None, batch_idx_2=None, B=1, edge_dist=None):

        # Pre-flatten indices for 2D operations
        src_flat = src.flatten()
        dst_flat = dst.flatten()

        # Geometric attention bias (§5.4): closer 3D neighbors get higher logits.
        # Shared by the edge and node attention steps; None when disabled.
        use_dist_bias = self.dist_bias_gamma > 0.0 and edge_dist is not None
        dist_bias = (-self.dist_bias_gamma * edge_dist) if use_dist_bias else None  # (E,)

        # 1. Update Edges (Rank 1) from Source Nodes (Rank 0)
        q_edge = self.W_q_edge(h1)                 # (E, dim)
        kv_src = self.W_kv_src(h0[src_flat])       # (E, dim*2) - Optimized: Flat 2D Indexing
        k_src, v_src = kv_src.chunk(2, dim=-1)     # 2 x (E, dim)

        attn_edge_logit = (q_edge * k_src).sum(dim=-1) / (self.dim ** 0.5)   # (E,)
        if dist_bias is not None:
            attn_edge_logit = attn_edge_logit + dist_bias
        if self.edge_attn_softmax:
            # Softmax over each source node's K outgoing edges (§5.3): a proper
            # attention normalisation instead of an element-wise sigmoid gate that
            # saturates toward 1.0 and degenerates to unweighted mean aggregation.
            attn_edge_exp = torch.exp(torch.clamp(attn_edge_logit, min=-10.0, max=10.0))
            attn_edge_sum = torch.zeros(h0.size(0), device=h0.device).index_add_(0, src_flat, attn_edge_exp)
            attn_edge = (attn_edge_exp / (attn_edge_sum[src_flat] + 1e-8)).unsqueeze(-1)  # (E, 1)
        else:
            attn_edge = torch.sigmoid(attn_edge_logit).unsqueeze(-1)

        h1 = self.norm_edge(h1 + self.edge_ffn(attn_edge * v_src))
        
        # 2. Update SSEs (Rank 2) using Node Neighborhoods (Rank 0)
        if h2.shape[0] > 0 and sse_map_0 is not None:
            q_sse = self.W_q_sse(h2)
            kv_node_sse = self.W_kv_node_sse(h0)
            k_node, v_node = kv_node_sse.chunk(2, dim=-1)
            
            q_sse_expanded = q_sse[sse_map_0]
            attn_sse = (q_sse_expanded * k_node).sum(dim=-1, keepdim=True) / (self.dim ** 0.5)
            
            attn_sse_exp = torch.exp(torch.clamp(attn_sse, min=-10.0, max=10.0))
            attn_sse_sum = torch.zeros(h2.size(0), 1, device=h0.device).index_add_(0, sse_map_0, attn_sse_exp)
            attn_sse_norm = attn_sse_exp / (attn_sse_sum[sse_map_0] + 1e-8)
            
            msg_to_sse = torch.zeros(h2.size(0), self.dim, device=h0.device).index_add_(0, sse_map_0, attn_sse_norm * v_node)
            
            h2 = self.norm_sse(h2 + self.sse_ffn(msg_to_sse))

        # 3. Refine Residues (Rank 0) from Edges (Rank 1)
        # Target node j queries its incoming edge e_ij
        q_node = self.W_q_node(h0)                 # (N, dim)
        kv_dst = self.W_kv_dst(h1)                 # (E, dim*2) - Optimized: No flatten(0,1)
        k_dst, v_dst = kv_dst.chunk(2, dim=-1)     # 2 x (N*K, dim)
        
        q_node_expanded = q_node[dst_flat]         # (E, dim)
        
        attn_node = (q_node_expanded * k_dst).sum(dim=-1, keepdim=True) / (self.dim ** 0.5)
        if dist_bias is not None:
            attn_node = attn_node + dist_bias.unsqueeze(-1)

        attn_node_exp = torch.exp(torch.clamp(attn_node, min=-10.0, max=10.0))
        attn_node_sum = torch.zeros(h0.size(0), 1, device=h0.device).index_add_(0, dst_flat, attn_node_exp)
        attn_node_norm = attn_node_exp / (attn_node_sum[dst_flat] + 1e-8)
        
        msg_to_node = torch.zeros(h0.size(0), self.dim, device=h0.device).index_add_(0, dst_flat, attn_node_norm * v_dst)

        if self.capture_attn:
            self.last_node_attn = attn_node_norm.detach().squeeze(-1)
            self.last_attn_src = src_flat.detach()
            self.last_attn_dst = dst_flat.detach()

        h0 = self.norm_node(h0 + self.node_ffn(msg_to_node))

        # 4. Pool to Rank 3 (Global Protein Embedding)
        ones_h0 = torch.ones_like(h0[:, :1])
        h0_sum = torch.zeros(B, h0.size(-1), device=h0.device).index_add_(0, batch_idx_0, h0)
        h0_count = torch.zeros(B, 1, device=h0.device).index_add_(0, batch_idx_0, ones_h0)
        h0_pool = h0_sum / h0_count.clamp(min=1)
        if h2.shape[0] > 0:
            h2_sum = torch.zeros(B, h2.size(-1), device=h2.device).index_add_(0, batch_idx_2, h2)
            h2_count = torch.zeros(B, 1, device=h2.device).index_add_(0, batch_idx_2, torch.ones_like(h2[:, :1]))
        else:
            h2_sum = torch.zeros_like(h0_pool)
            h2_count = torch.ones_like(h0_count)
        h2_pool = h2_sum / h2_count.clamp(min=1)

        # Per-layer global update. Detaching the pooled features (§5.7) lets h3 carry
        # forward-pass conditioning to the next layer while cutting the shortcut
        # gradient path that lets SupCon encode cluster identity in h3 directly.
        if self.detach_h3:
            glob_in = torch.cat([h3, h0_pool.detach(), h2_pool.detach()], dim=-1)
        else:
            glob_in = torch.cat([h3, h0_pool, h2_pool], dim=-1)
        h3 = self.norm_global(h3 + self.W_global(glob_in))

        return h0, h1, h2, h3

class AsymmetricTopoNet(nn.Module):
    """
    Asymmetric TopoNet
    Wraps the AsymmetricTopoAttentionLayer into a full network mirroring Topotein.
    """
    def __init__(self, scalar_dim=128, num_layers=4, dropout=0.1,
                 use_positional_encoding=True, use_residue_features=True,
                 use_3di_features=True, edge_attn_softmax=False,
                 dist_bias_gamma=0.0, detach_h3=False,
                 dist_encoding='sinusoidal', rbf_dim=16,
                 rbf_dmin=3.5, rbf_dmax=20.0):
        super().__init__()

        # Distance-encoding choice (plan_implementation §1, Modification 1). The
        # rank-1 edge scalar is the raw Cα–Cα distance plus an expansion of it:
        #   'sinusoidal' -> the stored 16-d Transformer encoding (original; ~half
        #                   its bands are dead over the 0–20 Å contact range).
        #   'rbf'        -> K Gaussian radial bases computed in-forward from the
        #                   raw distance (calibrated to the contact scale). No
        #                   re-lifting needed; the raw distance is already stored.
        # Default 'sinusoidal' preserves the original forward pass (and keeps old
        # checkpoints loadable); the training entrypoint opts into 'rbf'.
        assert dist_encoding in ('sinusoidal', 'rbf')
        self.dist_encoding = dist_encoding
        if dist_encoding == 'rbf':
            from rbf import GaussianRBF
            self.rbf = GaussianRBF(num_rbf=rbf_dim, d_min=rbf_dmin, d_max=rbf_dmax)
        edge_in = 1 + (rbf_dim if dist_encoding == 'rbf' else 16)

        # Shortcut mitigations (report 7). Each flag, when False, zeros the
        # corresponding Rank-0 input channel WITHOUT changing tensor dims, so
        # checkpoints stay compatible. Defaults preserve original behavior.
        #   use_positional_encoding=False -> kill the sequence-index signal that
        #       drives the sequential-attention shortcut (report 3.2).
        #   use_residue_features=False    -> drop residue identity (composition
        #       shortcut, report 2.2 / 7).
        #   use_3di_features=False        -> drop the Foldseek 3Di alphabet
        #       (benchmark shortcut, report 5).
        self.use_positional_encoding = use_positional_encoding
        self.use_residue_features = use_residue_features
        self.use_3di_features = use_3di_features

        self.rank0_emb = nn.Sequential(
            nn.Linear(64, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        self.rank1_emb = nn.Sequential(
            nn.Linear(edge_in, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        self.rank2_input_norm = nn.LayerNorm(12)
        self.rank2_emb = nn.Sequential(
            nn.Linear(12, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        self.rank3_input_norm = nn.LayerNorm(10)
        self.rank3_emb = nn.Sequential(
            nn.Linear(10, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        self.layers = nn.ModuleList([
            AsymmetricTopoAttentionLayer(
                scalar_dim, dropout,
                edge_attn_softmax=edge_attn_softmax,
                dist_bias_gamma=dist_bias_gamma,
                detach_h3=detach_h3,
            )
            for _ in range(num_layers)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(scalar_dim * 2, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.LayerNorm(scalar_dim)
        )

    def forward(self, features, return_nodes=False, return_repr=False):
        r0, r1, r2_feat, r3 = features['rank0'], features['rank1'], features['rank2_features'], features['rank3']
        device = r0['aa'].device
        
        batch_idx_0 = features.get('batch_idx_0', torch.zeros(r0['aa'].shape[0], dtype=torch.long, device=device))
        batch_idx_2 = features.get('batch_idx_2', torch.zeros(r2_feat.size(0), dtype=torch.long, device=device))
        B = r3['protein_size'].size(0) if 'protein_size' in r3 else 1

        aa = r0['aa'] if self.use_residue_features else torch.zeros_like(r0['aa'])
        di = r0['3di'] if self.use_3di_features else torch.zeros_like(r0['3di'])
        pe = r0['positional_encoding'] if self.use_positional_encoding else torch.zeros_like(r0['positional_encoding'])
        h0 = self.rank0_emb(torch.cat([aa, di, r0['phi_psi'], pe], dim=-1))
        # Rank-1 edge scalars: raw distance + its expansion (RBF or sinusoidal,
        # §1). RBF is computed in-forward from the stored raw distance; the
        # sinusoidal path reuses the stored distance_encoding. Flatten to (E, ·).
        if self.dist_encoding == 'rbf':
            dist_exp = self.rbf(r1['distance'])               # (N, K_edges, rbf_dim)
        else:
            dist_exp = r1['distance_encoding']                # (N, K_edges, 16)
        h1_raw = torch.cat([r1['distance'].unsqueeze(-1), dist_exp], dim=-1).flatten(0, 1)
        h1 = self.rank1_emb(h1_raw)
        h2 = self.rank2_emb(self.rank2_input_norm(r2_feat)) if r2_feat.numel() > 0 else torch.empty((0, 128), device=device)
        
        p_size = r3['protein_size'].view(B, 1).to(device)
        rog = r3['radius_of_gyration'].view(B, 1).to(device)
        desc = r3['global_shape_descriptors'].view(B, 5).to(device)
        eigen = r3['global_eigenvalues'].view(B, 3).to(device)
        h3_raw = torch.cat([p_size, rog, desc, eigen], dim=-1)
        h3 = self.rank3_emb(self.rank3_input_norm(h3_raw))
        
        sse_map_0 = features.get('sse_map_0')
        # Per-edge Ca-Ca distance for the geometric attention bias (§5.4). Flattened
        # to (E,) to match src/target.flatten(); zero for padded dummy edges (which
        # self-loop on dummy nodes, so they never reach real nodes).
        edge_dist = r1['distance'].flatten()
        for layer in self.layers:
            h0, h1, h2, h3 = layer(h0, h1, h2, h3, r1['source'], r1['target'], sse_map_0, batch_idx_0, batch_idx_2, B, edge_dist=edge_dist)
            
        h0_sum = torch.zeros(B, h0.size(-1), device=device).index_add_(0, batch_idx_0, h0)
        h0_count = torch.zeros(B, 1, device=device).index_add_(0, batch_idx_0, torch.ones_like(h0[:, :1]))
        h0_pool = h0_sum / h0_count.clamp(min=1)
        
        # `repr` is the readout representation h (pre-L2-normalization). Eval uses
        # normalize(h); the contrastive projection head (plan_impl_3 §3) consumes h.
        repr_ = self.output_head(torch.cat([h0_pool, h3], dim=-1))
        out = F.normalize(repr_, p=2, dim=-1)
        out_formatted = out if B > 1 else out.squeeze(0)

        if return_repr:
            return repr_ if B > 1 else repr_.squeeze(0)
        if return_nodes:
            return out_formatted, h0

        return out_formatted
