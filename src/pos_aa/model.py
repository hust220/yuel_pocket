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

        # Protein (N_ATOM_TYPES) + Joint (1) + Ligand (N_ATOM_TYPES) + Masks (4)
        self.in_node_nf = const.N_ATOM_TYPES + 1 + const.N_ATOM_TYPES + 4 

        # Edge: Dist (1) + Bond (1) + Masks (2)
        self.in_edge_nf = 1 + 1 + 2
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
        
        self.probe_gcl = GCL(hidden_nf)
        
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
            ligand_mask.to(dtype=TORCH_FLOAT),
            g.ndata['sas_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [N, 4]
        
        h = torch.cat([h, masks], dim=-1)
        
        # Concat masks to e
        edge_masks = torch.stack([
            g.edata['complex_mask'].to(dtype=TORCH_FLOAT),
            g.edata['probe_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [E, 2]
        
        e = torch.cat([e, edge_masks], dim=-1)
        
        # Filter Edges (Complex Only) for GNN message passing
        complex_edge_mask = g.edata['complex_mask'].bool()
        e_complex = e[complex_edge_mask]
        edge_index_complex = edge_index[:, complex_edge_mask]
        
        # Run GNN on full nodes but restricted edges
        # Probes will be isolated nodes in this pass
        h_out, e_out = self.gnn(h, edge_index_complex, e_complex)
        
        # Update Probes using Probe Edges
        probe_edge_mask = g.edata['probe_mask'].bool()
        e_probe = e[probe_edge_mask]
        edge_index_probe = edge_index[:, probe_edge_mask]
        
        # We only want to update detailed features for probes based on protein context
        # h_out currently contains updated protein features and raw probe features (zeros)
        # We run one GCL step to pull info from proteins to probes
        # Note: GCL updates all nodes involved in edges.
        # Must embed e_probe first as GCL expects hidden dimensions
        e_probe_emb = self.gnn.emb_edge(e_probe)
        h_out, _ = self.probe_gcl(h_out, edge_index_probe, e_probe_emb)
        
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
        
        # is_pocket/is_decoy now marks the SINGLE probe node per graph
        raw_is_pocket = g.ndata['is_pocket']
        raw_is_decoy = g.ndata['is_decoy']
        
        # Iterate over each graph using the helper method
        for graph_mask_bool in g.get_batch_masks():
            # Slice/Filter for current graph
            sub_pred = pred_logits[graph_mask_bool]
            sub_pocket = raw_is_pocket[graph_mask_bool]
            sub_decoy = raw_is_decoy[graph_mask_bool]
            
            # Find the probe indices
            pos_mask = (sub_pocket == 1)
            decoy_mask = (sub_decoy == 1)
            
            if pos_mask.any() and decoy_mask.any():
                # Should be exactly one positive and one decoy probe per graph
                pos_score = sub_pred[pos_mask].mean()
                decoy_score = sub_pred[decoy_mask].mean()
                
                # Margin Loss
                # We want pos_score > decoy_score + margin
                # Loss = ReLU(decoy_score - pos_score + margin)
                margin_loss = F.relu(decoy_score - pos_score + margin)
                
                total_loss = total_loss + margin_loss
                total_mean_pos += pos_score.item()
                total_mean_decoy += decoy_score.item()
                n_valid_graphs += 1
                
                if pos_score > decoy_score:
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
            ligand_mask.to(dtype=TORCH_FLOAT),
            g.ndata['sas_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [N, 4]
        
        h = torch.cat([h, masks], dim=-1)
        
        # Concat masks to e
        edge_masks = torch.stack([
            g.edata['complex_mask'].to(dtype=TORCH_FLOAT),
            g.edata['probe_mask'].to(dtype=TORCH_FLOAT)
        ], dim=-1) # [E, 2]
        
        e = torch.cat([e, edge_masks], dim=-1)
        
        # Filter Edges (Complex Only) for GNN message passing
        complex_edge_mask = g.edata['complex_mask'].bool()
        e_complex = e[complex_edge_mask]
        edge_index_complex = edge_index[:, complex_edge_mask]
        
        # Run GNN on full nodes but restricted edges
        h_out, e_out = self.gnn(h, edge_index_complex, e_complex)
        
        # Update Probes using Probe Edges
        probe_edge_mask = g.edata['probe_mask'].bool()
        e_probe = e[probe_edge_mask]
        edge_index_probe = edge_index[:, probe_edge_mask]
        
        e_probe_emb = self.gnn.emb_edge(e_probe)
        h_out, _ = self.probe_gcl(h_out, edge_index_probe, e_probe_emb)
        
        # Predict score
        pred_logits = self.pocket_head(h_out).squeeze(-1) # [N]
        
        return pred_logits

