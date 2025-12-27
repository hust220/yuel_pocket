import torch
import torch.nn as nn
import torch.nn.functional as F
from src import const
from src.egnn import EGNN
from src.const import TORCH_INT, TORCH_FLOAT

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        return torch.nn.SiLU()

class YuelPocket(nn.Module):
    def __init__(
        self,
        hidden_nf=64,
        activation='silu', 
        n_layers=3, 
        sin_embedding=True, 
        normalization_factor=1, 
        aggregation_method='sum',
        **kwargs
    ):
        super(YuelPocket, self).__init__()

        self.in_node_nf = const.N_RESIDUE_TYPES + const.N_ATOM_TYPES + 1 + 3 # +3 for masks (protein, joint, ligand)
        self.in_edge_nf = 1 + 1 + 1 + 1 + const.N_RDBOND_TYPES
        self.hidden_nf = hidden_nf
        self.out_node_nf = 1 
        self.out_edge_nf = 1

        if isinstance(activation, str):
            activation = get_activation(activation)

        # Use EGNN from src.egnn
        self.egnn = EGNN(
            c_h=hidden_nf,
            n_layers=n_layers,
            in_node_dim=self.in_node_nf,
            in_edge_dim=self.in_edge_nf,
            out_node_dim=self.out_node_nf,
            out_edge_dim=self.out_edge_nf,
        )

    def forward(self, g, training=False):
        # Get tensors from graph
        h = g.ndata['h']
        x = g.ndata['x']
        e = g.edata['e']
        edge_index = g.edge_index
        
        protein_mask = g.ndata['protein_mask']
        joint_mask = g.ndata['joint_mask']
        ligand_mask = g.ndata['ligand_mask']
        
        # Concat masks to h
        masks = torch.stack([
            protein_mask.to(dtype=TORCH_FLOAT), 
            joint_mask.to(dtype=TORCH_FLOAT), 
            ligand_mask.to(dtype=TORCH_FLOAT)
        ], dim=-1) # [N, 3]
        
        h = torch.cat([h, masks], dim=-1)
        
        # Prepare input x (x_in)
        x_in = x.clone()
        
        # Set Joint nodes to 0 
        x_in[joint_mask.bool()] = 0.0

        # Run EGNN
        # EGNN expects: x, h, edges, e
        x_out, h_out, e_out = self.egnn(x_in, h, edge_index, e)
        
        metrics = {}
        
        # Compute MSE Loss for Joint Nodes
        joint_indices = joint_mask.bool()
        pred_joint = x_out[joint_indices]
        true_joint = x[joint_indices]
        
        # MSE Loss
        loss = F.mse_loss(pred_joint, true_joint)
        metrics['loss'] = loss
        
        return metrics

    def sample_chain(self, *args, **kwargs):
        return None
