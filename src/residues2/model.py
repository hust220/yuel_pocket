import torch
import torch.nn as nn
import torch.nn.functional as F
from src import const
from src.egnn import GNN
from src.const import TORCH_INT, TORCH_FLOAT

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        return torch.nn.SiLU()

def dice_loss(logits, targets, smooth=1.0):
    """
    Computes the Dice loss.
    logits: [N]
    targets: [N]
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    dice = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1. - dice

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

        # in_node_nf = (const.N_RESIDUE_TYPES + 1) + (const.N_ATOM_TYPES + 1) + 3 (masks)
        self.in_node_nf = (const.N_RESIDUE_TYPES + 1) + (const.N_ATOM_TYPES + 1) + 3
        self.in_edge_nf = 5 # [dist, contact, bond, p_extra, l_extra]
        self.hidden_nf = hidden_nf

        if isinstance(activation, str):
            activation_fn = get_activation(activation)
        else:
            activation_fn = activation

        # Single GNN for all processing
        self.gnn = GNN(
            c_h=hidden_nf,
            n_layers=n_layers,
            in_node_dim=self.in_node_nf,
            in_edge_dim=self.in_edge_nf,
            out_node_dim=hidden_nf,
            out_edge_dim=1,
        )
        
        # Pocket Prediction Head (Residual Interaction)
        self.pocket_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation_fn,
            nn.Linear(hidden_nf, 1)
        )

        # Ligand Pairing Head (Global Comparison)
        self.pairing_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation_fn,
            nn.Linear(hidden_nf, 1)
        )

        # 移动平均 F1 缓冲区，用于动态开启 Dice Loss
        self.register_buffer('f1_ma', torch.tensor(0.0))

    def forward(self, g, training=False):
        h = g.ndata['h']
        e = g.edata['e']
        edge_index = g.edge_index
        
        prot_mask = g.ndata['protein_mask'].bool()
        lig_mask = g.ndata['ligand_mask'].bool()
        extra_mask = g.ndata['extra_mask'].bool()
        is_true_ligand = g.ndata['is_true_ligand'].bool()
        
        # Concat masks to node features
        m_t = torch.stack([
            g.ndata['protein_mask'].to(TORCH_FLOAT),
            g.ndata['ligand_mask'].to(TORCH_FLOAT),
            g.ndata['extra_mask'].to(TORCH_FLOAT)
        ], dim=-1)
        h_in = torch.cat([h, m_t], dim=-1)
        
        # 1. Run GNN on the whole batch
        h_out, _ = self.gnn(h_in, edge_index, e)
        
        # Each mask in get_batch_masks() represents one complete sample (1 protein + 51 ligands)
        subgraph_masks = g.get_batch_masks() 
        num_samples = len(subgraph_masks)
        
        loss_pocket = torch.tensor(0.0, device=h.device)
        loss_pairing = torch.tensor(0.0, device=h.device)
        
        metrics = {
            'acc': torch.tensor(0.0, device=h.device),
            'prec': torch.tensor(0.0, device=h.device),
            'rec': torch.tensor(0.0, device=h.device),
            'success_rate': torch.tensor(0.0, device=h.device)
        }
        n_samples_processed = 0

        for i in range(num_samples):
            mask = subgraph_masks[i]
            
            # Extract nodes for this sample
            h_sample = h_out[mask]
            p_mask_sample = prot_mask[mask]
            l_mask_sample = lig_mask[mask]
            e_mask_sample = extra_mask[mask]
            true_mask_sample = is_true_ligand[mask]
            
            # A. Protein Extra Node [1, hidden]
            v_p_extra = h_sample[p_mask_sample & e_mask_sample]
            
            # B. Ligands Extra Nodes [51, hidden]
            v_l_extras = h_sample[l_mask_sample & e_mask_sample]
            
            # C. Normal Protein Nodes [N_res, hidden]
            h_p_normal = h_sample[p_mask_sample & ~e_mask_sample]
            
            # D. True Ligand Extra Node [1, hidden]
            v_l_true_extra = h_sample[l_mask_sample & e_mask_sample & true_mask_sample]
            
            if v_p_extra.size(0) == 0 or v_l_extras.size(0) == 0 or h_p_normal.size(0) == 0 or v_l_true_extra.size(0) == 0:
                continue

            # ---------------------------------------------------------
            # 1. Pairing Loss (Contrastive/InfoNCE)
            # Compare Protein Extra with ALL Ligand Extras
            pairing_input = v_p_extra * v_l_extras # Broad-casting: [1, hidden] * [51, hidden] -> [51, hidden]
            pairing_logits = self.pairing_head(pairing_input).squeeze(-1) # [51]
            
            # Target is the index of the true ligand within v_l_extras
            # Use is_true_ligand and extra_mask to find the exact index in this sample
            l_extra_mask_sample = l_mask_sample & e_mask_sample
            true_extra_mask_local = is_true_ligand[mask] & l_extra_mask_sample
            
            # Find which position in v_l_extras the true ligand occupies
            # v_l_extras was filtered by l_extra_mask_sample
            target_idx = torch.where(true_extra_mask_local[l_extra_mask_sample])[0]
            
            if target_idx.size(0) == 0:
                continue
            
            temperature = 0.1
            lp_info = F.cross_entropy(pairing_logits.unsqueeze(0) / temperature, target_idx)
            loss_pairing = loss_pairing + lp_info
            
            # Success Rate: is true ligand score the max?
            if torch.argmax(pairing_logits) == target_idx[0]:
                metrics['success_rate'] += 1.0

            # ---------------------------------------------------------
            # 2. Pocket Loss (Residual classification)
            # Compare Normal Protein Nodes with TRUE Ligand Extra
            fused_pocket = h_p_normal * v_l_true_extra # Broad-casting: [N_res, hidden] * [1, hidden]
            pocket_logits = self.pocket_head(fused_pocket).squeeze(-1) # [N_res]
            
            targets_p = g.ndata['is_pocket'][mask][p_mask_sample & ~e_mask_sample]
            
            num_pos = targets_p.sum()
            num_neg = len(targets_p) - num_pos
            # A. Weighted BCE (Always on)
            pos_weight = (num_neg / (num_pos + 1e-6)).clamp(min=1.0, max=100.0)
            lbce = F.binary_cross_entropy_with_logits(pocket_logits, targets_p, pos_weight=pos_weight)
            
            # B. Conditional Dice Loss activation
            with torch.no_grad():
                probs_b = torch.sigmoid(pocket_logits)
                preds_b = (probs_b > 0.5).float()
                tp_b = (preds_b * targets_p).sum()
                fp_b = (preds_b * (1 - targets_p)).sum()
                fn_b = ((1 - preds_b) * targets_p).sum()
                batch_f1 = (2. * tp_b) / (2. * tp_b + fp_b + fn_b + 1e-6)
                # Update Moving Average (Smooth factor 0.9)
                self.f1_ma.copy_(0.9 * self.f1_ma + 0.1 * batch_f1)

            lp = lbce
            if self.f1_ma > 0.2:
                lp = lp + dice_loss(pocket_logits, targets_p)
                
            loss_pocket = loss_pocket + lp
            
            # Pocket Metrics
            probs = torch.sigmoid(pocket_logits)
            preds = (probs > 0.5).float()
            tp = (preds * targets_p).sum()
            fp = (preds * (1 - targets_p)).sum()
            fn = ((1 - preds) * targets_p).sum()
            metrics['acc'] += (preds == targets_p).float().mean()
            metrics['prec'] += (tp / (tp + fp + 1e-6))
            metrics['rec'] += (tp / (tp + fn + 1e-6))
            
            n_samples_processed += 1

        if n_samples_processed > 0:
            metrics['loss'] = (loss_pocket + loss_pairing) / n_samples_processed
            metrics['pocket_loss'] = loss_pocket / n_samples_processed
            metrics['pairing_loss'] = loss_pairing / n_samples_processed
            
            for k in ['acc', 'prec', 'rec', 'success_rate']:
                metrics[k] /= n_samples_processed
            
            prec, rec = metrics['prec'], metrics['rec']
            metrics['f1'] = 2 * (prec * rec) / (prec + rec + 1e-6)
        else:
            # Fallback to ensure loss is connected to model parameters
            metrics['loss'] = h_out.sum() * 0.0
            
        return metrics

    def sample_chain(self, g):
        """
        Inference method.
        Returns:
            all_pairing_probs: List of tensors, each [num_ligands] probabilities
            all_pocket_probs: List of tensors, each [num_residue_nodes] probabilities
        """
        h = g.ndata['h']
        e = g.edata['e']
        edge_index = g.edge_index
        
        prot_mask = g.ndata['protein_mask'].bool()
        lig_mask = g.ndata['ligand_mask'].bool()
        extra_mask = g.ndata['extra_mask'].bool()
        
        # Concat masks to node features
        m_t = torch.stack([
            g.ndata['protein_mask'].to(TORCH_FLOAT),
            g.ndata['ligand_mask'].to(TORCH_FLOAT),
            g.ndata['extra_mask'].to(TORCH_FLOAT)
        ], dim=-1)
        h_in = torch.cat([h, m_t], dim=-1)
        
        # 1. Run GNN
        h_out, _ = self.gnn(h_in, edge_index, e)
        
        subgraph_masks = g.get_batch_masks()
        all_pairing_probs = []
        all_pocket_probs = []
        
        for mask in subgraph_masks:
            h_sample = h_out[mask]
            pm = prot_mask[mask]
            lm = lig_mask[mask]
            em = extra_mask[mask]
            
            # Global Pairing (Protein Extra vs Ligand Extras)
            v_p_extra = h_sample[pm & em] # [1, H]
            v_l_extras = h_sample[lm & em] # [M, H]
            
            if v_p_extra.size(0) > 0 and v_l_extras.size(0) > 0:
                pairing_input = v_p_extra * v_l_extras # [M, H]
                pairing_logits = self.pairing_head(pairing_input).squeeze(-1) # [M]
                # During training we used cross_entropy(logits/0.1), so here we can use softmax or sigmoid.
                # If M=1, sigmoid is more informative of absolute confidence.
                pairing_prob = pairing_logits
                # pairing_prob = torch.sigmoid(pairing_logits / 0.1)
                all_pairing_probs.append(pairing_prob)
                
                # Pocket Prediction (Normal Residue nodes vs first/best ligand?)
                # We'll just use the first ligand's extra node for the pocket prediction
                h_p_normal = h_sample[pm & ~em] # [N_res_nodes, H]
                if h_p_normal.size(0) > 0:
                    fused_pocket = h_p_normal * v_l_extras[0:1] # [N_res_nodes, H]
                    pocket_logits = self.pocket_head(fused_pocket).squeeze(-1) # [N_res_nodes]
                    pocket_probs = torch.sigmoid(pocket_logits)
                    all_pocket_probs.append(pocket_probs)
                else:
                    all_pocket_probs.append(torch.tensor([]))
            else:
                all_pairing_probs.append(torch.tensor([]))
                all_pocket_probs.append(torch.tensor([]))
                
        return all_pairing_probs, all_pocket_probs
