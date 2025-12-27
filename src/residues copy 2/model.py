import torch
import torch.nn as nn
import torch.nn.functional as F
from src import const
from src.egnn import GNN
from src.const import TORCH_INT, TORCH_FLOAT
from src.residues.config import get_config

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

        # Use GNN from src.egnn
        # GNN(c_h, n_layers, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim)
        self.gnn = GNN(
            c_h=hidden_nf,
            n_layers=n_layers,
            in_node_dim=self.in_node_nf,
            in_edge_dim=self.in_edge_nf,
            out_node_dim=hidden_nf, # Output embedding for prediction head
            out_edge_dim=self.out_edge_nf,
        )
        
        self.pocket_head = nn.Linear(hidden_nf, 1)

    def forward(self, g, training=False):
        # Get tensors from graph
        h = g.ndata['h']
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
        
        # Run GNN
        # GNN expects: h, edges, e
        h_out, e_out = self.gnn(h, edge_index, e)
        
        # Predict is_pocket
        pred_logits = self.pocket_head(h_out).squeeze(-1) # [N]
        
        metrics = {}
        
        config = get_config()
        margin = config.get('contrastive_margin', 0.2)
        
        # Per-Graph Loss Calculation
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        total_mean_pos = 0.0
        total_mean_decoy = 0.0
        n_valid_graphs = 0
        n_success = 0
        
        # pred_logits is [N], aligned with g.ndata['h']
        raw_is_pocket = g.ndata['is_pocket']
        raw_is_decoy = g.ndata['is_decoy']
        raw_prot_mask = g.ndata['protein_mask']
        
        # Iterate over each graph using the helper method
        for graph_mask_bool in g.get_batch_masks():
            # Slice/Filter for current graph
            sub_pred = pred_logits[graph_mask_bool]
            sub_pocket = raw_is_pocket[graph_mask_bool]
            sub_decoy = raw_is_decoy[graph_mask_bool]
            sub_prot_mask_sliced = raw_prot_mask[graph_mask_bool].bool()
            
            # Filter for protein nodes only within this graph
            # Note: sub_pred and masks align because they are all node-level (within the graph)
            graph_pred = sub_pred[sub_prot_mask_sliced]
            graph_pocket = sub_pocket[sub_prot_mask_sliced]
            graph_decoy = sub_decoy[sub_prot_mask_sliced]
            
            # Safety check: must have at least one pocket residue and one decoy residue
            if graph_pocket.sum() > 0 and graph_decoy.sum() > 0:
                mean_pos = graph_pred[graph_pocket == 1].mean()
                mean_decoy = graph_pred[graph_decoy == 1].mean()
                
                # Margin Loss
                margin_loss = F.relu(mean_decoy - mean_pos + margin)
                
                total_loss = total_loss + margin_loss
                total_mean_pos += mean_pos.item()
                total_mean_decoy += mean_decoy.item()
                n_valid_graphs += 1
                
                # Check for success (simple classification accuracy per graph)
                if mean_pos > mean_decoy:
                    n_success += 1

        # Average over valid graphs
        if n_valid_graphs > 0:
            loss = total_loss / n_valid_graphs
            metrics['margin_loss'] = loss
            metrics['mean_pos'] = torch.tensor(total_mean_pos / n_valid_graphs, device=pred_logits.device)
            metrics['mean_decoy'] = torch.tensor(total_mean_decoy / n_valid_graphs, device=pred_logits.device)
            metrics['success_rate'] = torch.tensor(n_success / n_valid_graphs, device=pred_logits.device)
        else:
            loss = torch.tensor(0.0, device=pred_logits.device)
            metrics['margin_loss'] = loss
            metrics['mean_pos'] = torch.tensor(0.0, device=pred_logits.device)
            metrics['mean_decoy'] = torch.tensor(0.0, device=pred_logits.device)
            metrics['success_rate'] = torch.tensor(0.0, device=pred_logits.device)

        metrics['loss'] = loss
        
        return metrics

    def sample_chain(self, g, **kwargs):
        # Inference method to get raw scores
        # Get tensors from graph
        h = g.ndata['h']
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
        
        # Run GNN
        h_out, e_out = self.gnn(h, edge_index, e)
        
        # Predict score
        pred_logits = self.pocket_head(h_out).squeeze(-1) # [N]
        
        return pred_logits

