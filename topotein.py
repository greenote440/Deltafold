"""
Topotein Network Architecture
Implementation of the hierarchical topological message passing network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoteinInteractionLayer(nn.Module):
    """
    Hierarchical Message Passing Layer for Protein Combinatorial Complexes.
    Routes information between different topological ranks.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        
        # Node-to-Node (Rank 0 via Rank 1 Edges)
        self.msg_010 = nn.Linear(dim * 2, dim)
        self.upd_0 = nn.Linear(dim * 2, dim)
        
        # SSE-to-Node / Node-to-SSE (Rank 0 <-> Rank 2)
        self.msg_02 = nn.Linear(dim, dim)
        self.msg_20 = nn.Linear(dim, dim)
        self.upd_2 = nn.Linear(dim * 2, dim)
        
        # Global Update (Rank 3)
        self.upd_3 = nn.Linear(dim * 3, dim)
        
        self.norm0 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, h0, h1, h2, h3, src, dst, sse_map_0=None, nodes_per_sse=None, batch_idx_0=None, batch_idx_2=None, B=1):
        # 1. Rank 0 <-> Rank 1 Message Passing (Residue pairwise interactions)
        h0_src = h0[src] # (N, K, dim)
        msg_edge = F.silu(self.msg_010(torch.cat([h0_src, h1], dim=-1)))
        
        # Aggregate messages to targets with degree normalization to stabilize gradients
        dst_flat_idx = dst.flatten()
        dst_flat = dst_flat_idx.unsqueeze(-1).expand(-1, h0.shape[-1])
        msg_flat = msg_edge.flatten(0, 1)
        
        agg_0 = torch.zeros_like(h0).scatter_add_(0, dst_flat, msg_flat)
        deg_0 = torch.zeros(h0.size(0), 1, device=h0.device).scatter_add_(0, dst_flat_idx.unsqueeze(1), torch.ones_like(dst_flat_idx.unsqueeze(1).float()))
        agg_0 = agg_0 / deg_0.clamp(min=1)
        
        # 2. Rank 0 <-> Rank 2 Message Passing (Residues and Secondary Structures)
        if h2.shape[0] > 0 and sse_map_0 is not None:
            # Vectorized Node -> SSE aggregation
            msg_02_node = self.msg_02(h0)
            msg_02 = torch.zeros(h2.size(0), h2.size(1), device=h0.device).scatter_add_(
                0, sse_map_0.unsqueeze(1).expand(-1, h2.size(1)), msg_02_node
            )
            if nodes_per_sse is not None:
                msg_02 = msg_02 / nodes_per_sse.clamp(min=1)
            
            # Update SSE (Rank 2) state in one batch operation
            h2 = self.norm2(h2 + self.drop(F.silu(self.upd_2(torch.cat([h2, msg_02], dim=-1)))))
            
            # Vectorized SSE -> Node message passing
            agg_20 = self.msg_20(h2)[sse_map_0]
        else:
            agg_20 = torch.zeros_like(h0)
            
        # Finalize Node (Rank 0) update
        h0 = self.norm0(h0 + self.drop(F.silu(self.upd_0(torch.cat([agg_0, agg_20], dim=-1)))))
        
        # 3. Global Update (Rank 3)
        if batch_idx_0 is not None:
            h0_sum = torch.zeros(B, h0.size(-1), device=h0.device).scatter_add_(0, batch_idx_0.unsqueeze(1).expand(-1, h0.size(-1)), h0)
            h0_count = torch.zeros(B, 1, device=h0.device).scatter_add_(0, batch_idx_0.unsqueeze(1), torch.ones_like(h0[:, :1]))
            h0_pool = h0_sum / h0_count.clamp(min=1)
        else:
            h0_pool = h0.mean(dim=0, keepdim=True)
            
        if h2.shape[0] > 0:
            if batch_idx_2 is not None:
                h2_sum = torch.zeros(B, h2.size(-1), device=h2.device).scatter_add_(0, batch_idx_2.unsqueeze(1).expand(-1, h2.size(-1)), h2)
                h2_count = torch.zeros(B, 1, device=h2.device).scatter_add_(0, batch_idx_2.unsqueeze(1), torch.ones_like(h2[:, :1]))
                h2_pool = h2_sum / h2_count.clamp(min=1)
            else:
                h2_pool = h2.mean(dim=0, keepdim=True)
        else:
            h2_pool = torch.zeros_like(h0_pool)
        
        # Consolidate multiscale hierarchies into a single global state
        h3 = self.norm3(h3 + self.drop(F.silu(self.upd_3(torch.cat([h3, h0_pool, h2_pool], dim=-1)))))
        
        return h0, h1, h2, h3


class Topotein(nn.Module):
    """
    Topotein: Topological Deep Learning for Protein Representation Learning.
    Constructs hierarchical geometric representations from Protein Combinatorial Complexes.
    """
    def __init__(self, scalar_dim=128, num_layers=4, dropout=0.1,
                 use_positional_encoding=True, use_residue_features=True,
                 use_3di_features=True):
        super().__init__()

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

        # --- Embedding Modules for Multi-Rank Features ---
        
        # Rank 0 (Residues): aa (23) + 3di (21) + phi_psi (4) + positional (16) = 64
        self.rank0_emb = nn.Sequential(
            nn.Linear(64, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        # Rank 1 (Interactions): distance (1) + distance_encoding (16) = 17
        self.rank1_emb = nn.Sequential(
            nn.Linear(17, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        # Rank 2 (Secondary Structures): type (4) + shape_descriptors (5) + eigenvalues (3) = 12
        self.rank2_input_norm = nn.LayerNorm(12)
        self.rank2_emb = nn.Sequential(
            nn.Linear(12, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        # Rank 3 (Global Protein): size (1) + rog (1) + shape_descriptors (5) + eigenvalues (3) = 10
        self.rank3_input_norm = nn.LayerNorm(10)
        self.rank3_emb = nn.Sequential(
            nn.Linear(10, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim)
        )
        
        # Hierarchical Message Passing Layers (TCPNet Interaction Layers)
        self.layers = nn.ModuleList([
            TopoteinInteractionLayer(scalar_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output Head (Extracts Final Protein Embedding)
        self.output_head = nn.Sequential(
            nn.Linear(scalar_dim * 2, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.LayerNorm(scalar_dim)
        )

    def forward(self, features, return_nodes=False):
        """
        Forward pass for a single protein.
        Expects `features` dictionary directly from the TopoteinLifter script.
        """
        r0, r1, r2_feat, r3 = features['rank0'], features['rank1'], features['rank2_features'], features['rank3']
        device = r0['aa'].device
        
        batch_idx_0 = features.get('batch_idx_0', torch.zeros(r0['aa'].shape[0], dtype=torch.long, device=device))
        batch_idx_2 = features.get('batch_idx_2', torch.zeros(r2_feat.size(0), dtype=torch.long, device=device))
        B = r3['protein_size'].size(0) if 'protein_size' in r3 else 1

        # 1. Embed Features using respective linear projections
        aa = r0['aa'] if self.use_residue_features else torch.zeros_like(r0['aa'])
        di = r0['3di'] if self.use_3di_features else torch.zeros_like(r0['3di'])
        pe = r0['positional_encoding'] if self.use_positional_encoding else torch.zeros_like(r0['positional_encoding'])
        h0 = self.rank0_emb(torch.cat([aa, di, r0['phi_psi'], pe], dim=-1))
        h1 = self.rank1_emb(torch.cat([r1['distance'].unsqueeze(-1), r1['distance_encoding']], dim=-1))
        
        h2 = self.rank2_emb(self.rank2_input_norm(r2_feat)) if r2_feat.numel() > 0 else torch.empty((0, self.rank0_emb[0].out_features), device=device)
        
        p_size = r3['protein_size'].view(B, 1).to(device)
        rog = r3['radius_of_gyration'].view(B, 1).to(device)
        desc = r3['global_shape_descriptors'].view(B, 5).to(device)
        eigen = r3['global_eigenvalues'].view(B, 3).to(device)
        h3_raw = torch.cat([p_size, rog, desc, eigen], dim=-1)
        h3 = self.rank3_emb(self.rank3_input_norm(h3_raw))
        
        # 2. Hierarchical Message Passing
        sse_map_0 = features.get('sse_map_0')
        nodes_per_sse = features.get('nodes_per_sse')
        for layer in self.layers:
            h0, h1, h2, h3 = layer(h0, h1, h2, h3, r1['source'], r1['target'], sse_map_0, nodes_per_sse, batch_idx_0, batch_idx_2, B)
            
        # 3. Final Readout
        h0_sum = torch.zeros(B, h0.size(-1), device=device).scatter_add_(0, batch_idx_0.unsqueeze(1).expand(-1, h0.size(-1)), h0)
        h0_count = torch.zeros(B, 1, device=device).scatter_add_(0, batch_idx_0.unsqueeze(1), torch.ones_like(h0[:, :1]))
        h0_pool = h0_sum / h0_count.clamp(min=1)
        
        out = self.output_head(torch.cat([h0_pool, h3], dim=-1))
        out_formatted = out if B > 1 else out.squeeze(0)
        
        if return_nodes:
            return out_formatted, h0
            
        return out_formatted