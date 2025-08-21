import math
import numpy as np
import torch
import torch.nn as nn

from src import utils
from pdb import set_trace
from src import const

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff # (n_edges,1), (n_edges,n_feats)

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class GCL(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, in_edge_dim, out_edge_dim,
                 hidden_dim, edge_att_dim, node_att_dim, 
                 normalization_factor, aggregation_method, activation,
                 attention=False, normalization=None):
        super(GCL, self).__init__()

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_node_dim * 2 + in_edge_dim + edge_att_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_edge_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(out_edge_dim + in_node_dim + node_att_dim, hidden_dim),
#            nn.BatchNorm1d(hidden_nf),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_node_dim),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_feat, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target, edge_feat], dim=1)
        else:
            out = torch.cat([source, target, edge_feat, edge_attr], dim=1)

        out = edge_feat + self.edge_mlp(out)

        if edge_mask is not None:
            out = out * edge_mask

        return out

    def node_model(self, h, edge_index, edge_feat, node_attr, node_mask):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)

        out = h + self.node_mlp(agg)

        if node_mask is not None:
            out = out * node_mask

        return out

    def forward(self, h, edge_index, edge_feat, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        # print('edge_index.dtype', edge_index.dtype)
        # print('h.dtype', h.dtype)
        # print('edge_feat.dtype', edge_feat.dtype)
        # print('edge_attr.dtype', edge_attr.dtype)
        # print('edge_mask.dtype', edge_mask.dtype)
        # print('h.shape', h.shape)
        # print('row.max()', row.max())
        # print('col.max()', col.max())
        edge_feat = self.edge_model(h[row], h[col], edge_feat, edge_attr, edge_mask)
        h = self.node_model(h, edge_index, edge_feat, node_attr, node_mask)
        return h, edge_feat


def merge_edges(edge_index, edge_attr, edge_mask, node_mask):
    # edge_index: (b,n_edges,2)
    # edge_attr: (b,n_edges,n_feats)
    # edge_mask: (b,n_edges,1)
    # node_mask: (b,n_nodes,1)
    batch_size, n_edges, n_feats = edge_attr.shape
    
    n_nodes = node_mask.squeeze(-1).sum(1)
    n_nodes = torch.cumsum(n_nodes, dim=0)
    n_nodes = torch.cat([torch.zeros(1, dtype=n_nodes.dtype, device=n_nodes.device), n_nodes[:-1]])
    n_nodes = n_nodes.view(-1,1,1)

    edge_index = edge_index + n_nodes
    edge_index = edge_index.view(-1,2)
    edge_index = edge_index.t()

    edge_attr = edge_attr.view(-1, n_feats)

    edge_mask = edge_mask.view(-1, 1)

    return edge_index, edge_attr, edge_mask

class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, out_edge_nf=0, out_node_nf=0, 
                 device='cpu', activation=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(GNN, self).__init__()

        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.out_edge_nf = out_edge_nf

        self.edge_att_dim = in_edge_nf
        self.node_att_dim = 0

        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)

        if out_node_nf != 0:
            self.embedding_node_out = nn.Linear(self.hidden_nf, out_node_nf)
        if out_edge_nf != 0:
            self.embedding_edge_out = nn.Linear(self.hidden_nf, out_edge_nf)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                in_node_dim=self.hidden_nf, out_node_dim=self.hidden_nf,
                in_edge_dim=self.hidden_nf, out_edge_dim=self.hidden_nf,
                hidden_dim=self.hidden_nf, 
                edge_att_dim=self.edge_att_dim, node_att_dim=self.node_att_dim,
                activation=activation, attention=attention,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method))
            
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, edge_index, edge_attr, node_mask, edge_mask):
        # merge batches
        # h (b,n_nodes,n_feats) => (b*n_nodes,n_feats)
        # node_mask (b,n_nodes,1) => (b*n_nodes,1)
        # edge_mask (b,n_nodes,1) => (b*n_nodes,1)

        # print('h.shape', h.shape)
        batch_size, _, node_nf = h.shape
        h = h.view(-1, node_nf)
        edge_index, edge_attr, edge_mask = merge_edges(edge_index, edge_attr, edge_mask, node_mask)
        node_mask = node_mask.view(-1, 1)

        if edge_mask is not None:
           edge_mask = edge_mask.view(-1, 1)

        h = self.embedding_node(h)
        edge_feat = self.embedding_edge(edge_attr)
        for i in range(0, self.n_layers):
            h, edge_feat = self._modules["gcl_%d" % i](h, edge_index, edge_feat, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        if edge_mask is not None:
            edge_feat = edge_feat * edge_mask

        if self.out_node_nf != 0:
            h = self.embedding_node_out(h)
            h = h.view(batch_size, -1, self.out_node_nf) # (b, n_nodes, out_node_nf)

        if self.out_edge_nf != 0:
            edge_feat = self.embedding_edge_out(edge_feat)
            edge_feat = edge_feat.view(batch_size, -1, self.out_edge_nf) # (b, n_edges, out_edge_nf)

        return h, edge_feat

