import torch
import torch.nn as nn

class SumOuterAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SumOuterAttention, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.context_gate = nn.Linear(hidden_dim, 1)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.local_proj = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        
        # Output projection for the outer product result (dim * dim -> dim)
        self.out_proj = nn.Linear(hidden_dim * hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [Batch, SeqLen, Dim]
        """
        residual = x
        x = self.norm(x)
        b, n, c = x.shape
        
        # 1. Global Context
        # Compute gating weights and values
        gate = self.context_gate(x) # [B, N, 1]
        val = self.context_proj(x)  # [B, N, C]
        
        # Sum Pooling weighted by gate
        # Sum over sequence dimension (dim=1)
        global_context = torch.sum(val * gate, dim=1) / (n + 1e-6) # [B, C]

        # 2. Local Features
        local_feat = self.local_proj(x) # [B, N, C]
        
        # 3. Outer Product Interaction
        # Compute outer product between local features and global context
        # [B, N, i] x [B, j] -> [B, N, i, j]
        outer = torch.einsum('bni, bj -> bnij', local_feat, global_context)
        
        # Flatten to [B, N, C*C]
        outer_flat = outer.reshape(b, n, c * c)
        
        # 4. Project back
        h = self.out_proj(outer_flat)

        return residual + h, global_context

class SumOuterModel(nn.Module):
    def __init__(self, hidden_dim, n_layers, in_node_dim, out_node_dim):
        super(SumOuterModel, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.layers = nn.ModuleList([
            SumOuterAttention(hidden_dim) for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(hidden_dim, out_node_dim)

    def forward(self, x):
        # x: [B, N, in_node_dim]
        h = self.embedding(x)

        for layer in self.layers:
            h, global_context = layer(h)

        h = self.head(h)

        return h, global_context
