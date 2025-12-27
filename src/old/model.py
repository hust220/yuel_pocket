import torch
import torch.nn as nn
import torch.nn.functional as F
from src import const
from src.old.gnn import GNN as OldGNN
from src.const import TORCH_INT, TORCH_FLOAT

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        return torch.nn.SiLU() # Default

class GNN_Flat(OldGNN):
    """
    Wrapper around OldGNN to support flat batch input (PyG style) 
    instead of dense batch input (B, N, F).
    """
    def __init__(self, *args, **kwargs):
        # Force device='cpu' to prevent OldGNN from self.to(device) prematurely.
        # Lightning will handle device movement.
        kwargs['device'] = 'cpu'
        super().__init__(*args, **kwargs)

    def forward(self, h, edge_index, edge_attr, node_mask=None, edge_mask=None):
        # Skip batch merging logic since data is already flat
        
        # h: (N_total, in_node_nf)
        # edge_index: (2, E_total)
        # edge_attr: (E_total, in_edge_nf)
        
        # Determine number of nodes (used in segments) implicitly by h shape
        
        h = self.embedding_node(h)
        edge_feat = self.embedding_edge(edge_attr)
        
        for i in range(0, self.n_layers):
            # GCL forward expects: h, edge_index, edge_feat, edge_attr, ...
            h, edge_feat = self._modules["gcl_%d" % i](
                h, 
                edge_index, 
                edge_feat, 
                edge_attr=edge_attr, 
                node_mask=node_mask, 
                edge_mask=edge_mask
            ) # Output h is updated hidden state
            
        # Post-processing
        if self.out_node_nf != 0:
            h = self.embedding_node_out(h) # (N_total, out_node_nf)
            
        if self.out_edge_nf != 0:
            edge_feat = self.embedding_edge_out(edge_feat)

        return h, edge_feat

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

        self.in_node_nf = const.N_RESIDUE_TYPES + const.N_ATOM_TYPES + 1 + 3 # +3 for masks
        self.in_edge_nf = 1 + 1 + 1 + 1 + const.N_RDBOND_TYPES
        self.hidden_nf = hidden_nf
        self.out_node_nf = 1 # Predict 1 val (is_pocket logit)

        # Use GNN_Flat
        self.gnn = GNN_Flat(
            in_node_nf=self.in_node_nf,
            in_edge_nf=self.in_edge_nf,
            hidden_nf=hidden_nf,
            out_node_nf=self.out_node_nf,
            n_layers=n_layers,
            activation=get_activation(activation),
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method
        )

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
        # GNN_Flat expects: h, edge_index, edge_attr
        # Output h_out is (N, 1) because out_node_nf=1
        pred_logits, _ = self.gnn(h, edge_index, e)
        pred_logits = pred_logits.squeeze(-1) # [N]
        
        metrics = {}
        
        # Calculate Loss
        # Only on Protein Nodes
        prot_indices = protein_mask.bool()
        
        pred_pocket = pred_logits[prot_indices]
        true_pocket = g.ndata['is_pocket'][prot_indices]
        
        # Handle class imbalance
        num_pos = true_pocket.sum()
        num_neg = len(true_pocket) - num_pos
        pos_weight = (num_neg / (num_pos + 1e-6)).clamp(min=1.0)
        
        # BCE Loss
        loss = F.binary_cross_entropy_with_logits(pred_pocket, true_pocket, pos_weight=pos_weight)
        
        metrics['loss'] = loss
        
        # Metrics
        probs = torch.sigmoid(pred_pocket)
        preds = (probs > 0.5).float()
        
        tp = (preds * true_pocket).sum()
        fp = (preds * (1 - true_pocket)).sum()
        fn = ((1 - preds) * true_pocket).sum()
        
        accuracy = (preds == true_pocket).float().mean()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        metrics['acc'] = accuracy
        metrics['prec'] = precision
        metrics['rec'] = recall
        metrics['f1'] = f1
        metrics['num_pos'] = num_pos
        metrics['num_prot'] = torch.tensor(len(true_pocket), dtype=TORCH_FLOAT)
        metrics['num_lig'] = ligand_mask.sum()
        metrics['f1'] = f1
        
        return metrics

    def sample_chain(self, *args, **kwargs):
        return None
