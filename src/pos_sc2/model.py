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
        n_layers=3, 
        sin_embedding=True, 
        normalization_factor=1, 
        aggregation_method='sum',
        **kwargs
    ):
        super(YuelPocket, self).__init__()

        # Protein (const.N_RESIDUE_TYPES + 1 for BB) + Ligand (const.N_ATOM_TYPES) + Masks (3)
        self.in_node_nf = (const.N_RESIDUE_TYPES + 1) + const.N_ATOM_TYPES + 3

        # Edge: [Dist_Expanded (20), Bond (1)] + Masks (2)
        self.in_edge_nf = 20 + 1 + 2
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
            bi_directional=False
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
        
        # Run GNN on whole graph
        h_out, _ = self.gnn(h, edge_index, e)
        pred_logits = self.pocket_head(h_out).squeeze(-1)
        
        # 5. Loss and Metrics
        metrics = {}
        config = get_config()
        margin = config.get('contrastive_margin', 1.0)
        
        total_loss = torch.tensor(0.0, device=h.device)
        total_mean_pos, total_mean_decoy1, total_mean_decoy2 = 0.0, 0.0, 0.0
        n_valid_graphs, n_success = 0, 0
        
        raw_is_pocket = g.ndata['is_pocket']
        raw_is_decoy = g.ndata['is_decoy']
        raw_is_decoy2 = g.ndata['is_decoy2']

        for i, graph_mask_bool in enumerate(g.get_batch_masks()):
            sub_pred = pred_logits[graph_mask_bool]
            sub_pocket = raw_is_pocket[graph_mask_bool]
            sub_decoy = raw_is_decoy[graph_mask_bool]
            sub_decoy2 = raw_is_decoy2[graph_mask_bool]
            
            # Identify samples
            pos_m = (sub_pocket == 1)
            dec1_m = (sub_decoy == 1)
            dec2_m = (sub_decoy2 == 1)
            
            if pos_m.any() and (dec1_m.any() or dec2_m.any()):
                pos_s = sub_pred[pos_m].mean()
                
                graph_loss = torch.tensor(0.0, device=h.device)
                if dec1_m.any():
                    dec1_s = sub_pred[dec1_m].mean()
                    graph_loss = graph_loss + F.relu(dec1_s - pos_s + margin)
                    total_mean_decoy1 += dec1_s.item()
                if dec2_m.any():
                    dec2_s = sub_pred[dec2_m].mean()
                    graph_loss = graph_loss + F.relu(dec2_s - pos_s + margin)
                    total_mean_decoy2 += dec2_s.item()
                    
                total_loss = total_loss + graph_loss
                total_mean_pos += pos_s.item()
                n_valid_graphs += 1
                
                # Success Rate (using max/min for strictness or mean for consistency)
                # Here using mean comparison to match pos_sc
                if dec1_m.any() and dec2_m.any():
                    if pos_s > dec1_s and pos_s > dec2_s:
                        n_success += 1
                elif dec1_m.any() and pos_s > dec1_s:
                    n_success += 1
                elif dec2_m.any() and pos_s > dec2_s:
                    n_success += 1

        if n_valid_graphs > 0:
            loss = total_loss / n_valid_graphs
            metrics.update({
                'margin_loss': loss,
                'mean_pos': torch.tensor(total_mean_pos / n_valid_graphs, device=h.device),
                'mean_decoy1': torch.tensor(total_mean_decoy1 / n_valid_graphs, device=h.device),
                'mean_decoy2': torch.tensor(total_mean_decoy2 / n_valid_graphs, device=h.device),
                'success_rate': torch.tensor(n_success / n_valid_graphs, device=h.device),
                'loss': loss
            })
        else:
            loss = pred_logits.sum() * 0.0
            metrics.update({'margin_loss': loss, 'loss': loss, 'success_rate': torch.tensor(0.0, device=h.device)})
        return metrics

    def sample_chain(self, g, **kwargs):
        h = g.ndata['h']
        e = g.edata['e']
        edge_index = g.edge_index
        
        protein_mask = g.ndata['protein_mask']
        ligand_mask = g.ndata['ligand_mask']
        
        masks = torch.stack([
            protein_mask.to(dtype=TORCH_FLOAT), 
            ligand_mask.to(dtype=TORCH_FLOAT),
            g.ndata['sas_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [N, 3]
        
        h = torch.cat([h, masks], dim=-1)
        
        # Concat masks to e
        edge_masks = torch.stack([
            g.edata['complex_mask'].to(dtype=TORCH_FLOAT),
            g.edata['probe_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [E, 2]
        
        e = torch.cat([e, edge_masks], dim=-1)
        
        # Run GNN on whole graph
        h_out, e_out = self.gnn(h, edge_index, e)
        
        # Predict score
        pred_logits = self.pocket_head(h_out).squeeze(-1) # [N]
        return pred_logits

