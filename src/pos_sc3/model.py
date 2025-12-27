import torch
import torch.nn as nn
import torch.nn.functional as F
from src import const
from src.egnn import GNN, GCL
from src.const import TORCH_INT, TORCH_FLOAT
from .config import get_config

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
        n_layers=16, 
        sin_embedding=True, 
        normalization_factor=1, 
        aggregation_method='sum',
        **kwargs
    ):
        super(YuelPocket, self).__init__()
        config = get_config()
        n_layers = config.get('n_layers', n_layers)

        # Protein (const.N_RESIDUE_TYPES + 1 for BB) + Ligand (const.N_ATOM_TYPES) + Masks (3)
        self.in_node_nf = (const.N_RESIDUE_TYPES + 1) + const.N_ATOM_TYPES + 3

        # Edge: [Dist, Bond] (2) + Masks (2)
        self.in_edge_nf = 2 + 2
        self.hidden_nf = hidden_nf
        self.out_node_nf = 1 
        self.out_edge_nf = 1

        if isinstance(activation, str):
            activation = get_activation(activation)

        # Use GNN from src.egnn
        self.gnn = GNN(
            c_h=hidden_nf,
            n_layers=n_layers,
            in_node_dim=self.in_node_nf,
            in_edge_dim=self.in_edge_nf,
            out_node_dim=hidden_nf, 
            out_edge_dim=self.out_edge_nf,
        )
        
        self.pocket_head = nn.Linear(hidden_nf, 1)

    def forward(self, g, training=False):
        # Get tensors from graph
        h = g.ndata['h']
        e = g.edata['e']
        edge_index = g.edge_index
        
        protein_mask = g.ndata['protein_mask']
        ligand_mask = g.ndata['ligand_mask']
        sas_mask = g.ndata['sas_mask']
        
        # Concat masks to h
        masks = torch.stack([
            protein_mask.to(dtype=TORCH_FLOAT), 
            ligand_mask.to(dtype=TORCH_FLOAT),
            sas_mask.to(dtype=TORCH_FLOAT)
        ], dim=-1) # [N, 3]
        
        h = torch.cat([h, masks], dim=-1)
        
        # Concat masks to e
        edge_masks = torch.stack([
            g.edata['complex_mask'].to(dtype=TORCH_FLOAT),
            g.edata['probe_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [E, 2]
        
        e = torch.cat([e, edge_masks], dim=-1)
        
        # Run GNN on the whole graph (Disconnected subgraphs will process independently)
        h_out, _ = self.gnn(h, edge_index, e)
        
        # Predict score
        pred_logits = self.pocket_head(h_out).squeeze(-1) # [N]
        
        metrics = {}
        config = get_config()
        margin = config.get('contrastive_margin', 1.0)
        
        # Per-Graph Loss Calculation
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        total_mean_pos = 0.0
        total_mean_decoy = 0.0
        n_valid_graphs = 0
        n_success = 0
        
        raw_is_pocket = g.ndata['is_pocket']
        raw_is_decoy = g.ndata['is_decoy']
        raw_is_decoy2 = g.ndata['is_decoy2']
        
        for graph_mask_bool in g.get_batch_masks():
            sub_pred = pred_logits[graph_mask_bool]
            sub_pocket = raw_is_pocket[graph_mask_bool]
            sub_decoy = raw_is_decoy[graph_mask_bool]
            sub_decoy2 = raw_is_decoy2[graph_mask_bool]
            
            pos_mask = (sub_pocket == 1)
            decoy_mask = (sub_decoy == 1)
            decoy2_mask = (sub_decoy2 == 1)
            
            if pos_mask.any():
                # InfoNCE implementation:
                # Target: Pocket logit should be the highest among all probes in the system
                pos_logit = sub_pred[pos_mask] # Usually [1]
                
                neg_m = (sub_decoy == 1) | (sub_decoy2 == 1)
                if neg_m.any():
                    neg_logits = sub_pred[neg_m] # Up to [11]
                    
                    # Combine: [Pos, Neg1, Neg2, ...]
                    combined_logits = torch.cat([pos_logit, neg_logits], dim=0)
                    
                    # Apply temperature scaling
                    tau = config.get('temperature', 0.1)
                    combined_logits = combined_logits / tau
                    
                    # Target is index 0 (the pocket)
                    target = torch.tensor([0], device=combined_logits.device)
                    
                    graph_loss = F.cross_entropy(combined_logits.unsqueeze(0), target)
                    total_loss = total_loss + graph_loss
                    
                    # Metrics
                    total_mean_pos += pos_logit.mean().item()
                    max_neg_score = neg_logits.max().item()
                    if pos_logit.mean() > max_neg_score:
                        n_success += 1
                    n_valid_graphs += 1
                else:
                    # Fallback if no decoys: trivial success
                    n_success += 1
                    n_valid_graphs += 1
  
        if n_valid_graphs > 0:
            loss = total_loss / n_valid_graphs
            metrics['margin_loss'] = loss
            metrics['mean_pos'] = torch.tensor(total_mean_pos / n_valid_graphs, device=pred_logits.device)
            metrics['success_rate'] = torch.tensor(n_success / n_valid_graphs, device=pred_logits.device)
        else:
            loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
            metrics['margin_loss'] = loss
            metrics['success_rate'] = torch.tensor(0.0, device=pred_logits.device)

        metrics['loss'] = loss
        return metrics

    def sample_chain(self, g, **kwargs):
        h = g.ndata['h']
        e = g.edata['e']
        edge_index = g.edge_index
        
        protein_mask = g.ndata['protein_mask']
        ligand_mask = g.ndata['ligand_mask']
        sas_mask = g.ndata['sas_mask']
        
        masks = torch.stack([
            protein_mask.to(dtype=TORCH_FLOAT), 
            ligand_mask.to(dtype=TORCH_FLOAT),
            sas_mask.to(dtype=TORCH_FLOAT)
        ], dim=-1) # [N, 3]
        
        h = torch.cat([h, masks], dim=-1)
        
        # Concat masks to e
        edge_masks = torch.stack([
            g.edata['complex_mask'].to(dtype=TORCH_FLOAT),
            g.edata['probe_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [E, 2]
        
        e = torch.cat([e, edge_masks], dim=-1)
        
        # Run GNN on whole graph
        h_out, _ = self.gnn(h, edge_index, e)
        
        # Predict score
        pred_logits = self.pocket_head(h_out).squeeze(-1) # [N]
        # Return only probe scores (where sas_mask == 1)
        sas_mask = g.ndata['sas_mask'].bool()
        return pred_logits[sas_mask]

