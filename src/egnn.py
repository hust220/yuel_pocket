import torch
import torch.nn as nn

class GCL(nn.Module):
    def __init__(self, c_h, bi_directional=True):
        super(GCL, self).__init__()
        self.bi_directional = bi_directional

        self.c_h = c_h
        
        self.layer_norm_h = nn.LayerNorm(c_h)
        self.layer_norm_e = nn.LayerNorm(c_h)
        
        self.gate_node = nn.Linear(c_h, c_h)
        self.proj_src = nn.Linear(c_h, c_h)
        self.proj_dst = nn.Linear(c_h, c_h)
        self.proj_edge = nn.Linear(c_h, c_h)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(c_h, c_h),
            nn.SiLU(),
            nn.Linear(c_h, c_h),
        )
        
        self.edge2node = nn.Linear(c_h, c_h)
        self.gate_edge = nn.Linear(c_h, c_h)
        self.proj_node = nn.Linear(c_h, c_h)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(c_h, c_h),
            nn.SiLU(),
            nn.Linear(c_h, c_h),
        )

    def forward(self, h, edges, e):
        h = self.layer_norm_h(h)
        e = self.layer_norm_e(e)
        
        src, dst = edges

        gate = self.gate_node(h)
        h_src = self.proj_src(h) * gate
        h_dst = self.proj_dst(h) * gate
        h_src = h_src[src]
        h_dst = h_dst[dst]

        e = e + self.edge_mlp(self.proj_edge(e) * (h_src + h_dst) / 2.0)

        edge_msg = self.edge2node(e)
        edge_gate = self.gate_edge(e)
        edge_msg = edge_msg * edge_gate

        shape = (h.size(0), edge_msg.size(1))
        agg = edge_msg.new_zeros(shape)
        
        dst_idx = dst[:, None].expand(-1, edge_msg.size(1))
        
        agg.scatter_add_(0, dst_idx, edge_msg)
        if self.bi_directional:
            src_idx = src[:, None].expand(-1, edge_msg.size(1))
            agg.scatter_add_(0, src_idx, edge_msg)
        
        norm = edge_msg.new_zeros(shape)
        ones = edge_msg.new_ones(edge_msg.shape)
        norm.scatter_add_(0, dst_idx, ones)
        if self.bi_directional:
            norm.scatter_add_(0, src_idx, ones)
        norm[norm == 0] = 1

        agg = agg / (norm + 1e-6)
        
        h = h + self.node_mlp(self.proj_node(h) * agg)

        return h, e

class GNN(nn.Module):
    def __init__(self, c_h, n_layers, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, bi_directional=True):
        super(GNN, self).__init__()

        self.c_h = c_h
        self.n_layers = n_layers
        self.bi_directional = bi_directional
        
        self.emb_node = nn.Sequential(
            nn.Linear(in_node_dim, c_h * 2),
            nn.SiLU(),
            nn.Linear(c_h * 2, c_h),
        )
        self.emb_edge = nn.Sequential(
            nn.Linear(in_edge_dim, c_h * 2),
            nn.SiLU(),
            nn.Linear(c_h * 2, c_h),
        )
        
        self.layers = nn.ModuleList([GCL(c_h=c_h, bi_directional=bi_directional) for _ in range(n_layers)])
        
        self.out_node = nn.Linear(c_h, out_node_dim)
        self.out_edge = nn.Linear(c_h, out_edge_dim)

    def forward(self, h, edges, e):
        h = self.emb_node(h)
        e = self.emb_edge(e)

        for layer in self.layers:
            h, e = layer(h, edges, e)

        h = self.out_node(h)
        e = self.out_edge(e)

        return h, e



class EGNN(nn.Module):
    def __init__(self, c_h, n_layers, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, bi_directional=True):
        super(EGNN, self).__init__()

        self.c_h = c_h
        self.n_layers = n_layers
        self.bi_directional = bi_directional
        
        self.emb_node = nn.Sequential(
            nn.Linear(in_node_dim, c_h * 2),
            nn.SiLU(),
            nn.Linear(c_h * 2, c_h),
        )
        self.emb_edge = nn.Sequential(
            nn.Linear(in_edge_dim + 1, c_h * 2),
            nn.SiLU(),
            nn.Linear(c_h * 2, c_h),
        )
        
        self.layers = nn.ModuleList([GCL(c_h=c_h, bi_directional=bi_directional) for _ in range(n_layers)])
        
        self.out_node = nn.Linear(c_h, out_node_dim)
        self.out_edge = nn.Linear(c_h, out_edge_dim)
        self.dist_out = nn.Linear(c_h, 1)

    def forward(self, x, h, edges, e):
        src, dst = edges
        r_ij = x[src] - x[dst] # (n_edges, 3)
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
        r_ij = r_ij / (d_ij + 1e-8) # (n_edges, 3)

        h = self.emb_node(h)
        e = self.emb_edge(torch.cat([e, d_ij], dim=-1))

        for layer in self.layers:
            h, e = layer(h, edges, e)

        dist = self.dist_out(e) # (n_edges, 1)
        dist = dist * r_ij # (n_edges, 3)
        dst_idx = dst[:, None].expand(-1, dist.size(1))
        x.scatter_add_(0, dst_idx, -dist) # (n_nodes, 3)
        if self.bi_directional:
            src_idx = src[:, None].expand(-1, dist.size(1))
            x.scatter_add_(0, src_idx, dist) # (n_nodes, 3)

        h = self.out_node(h)
        e = self.out_edge(e)

        return x, h, e

