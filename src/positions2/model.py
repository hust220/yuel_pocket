import torch
import torch.nn as nn
import torch.nn.functional as F
from src import const
from src.sumop import SumOuterModel
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
        **kwargs
    ):
        super(YuelPocket, self).__init__()

        self.hidden_nf = hidden_nf
        
        # Dimensions
        # Prot: [feat(46)]
        # Mol: [feat(17)]
        # SAS: [PE(6)]
        
        self.prot_raw_dim = const.N_RESIDUE_TYPES + 6 
        self.mol_raw_dim = const.N_ATOM_TYPES
        self.sas_raw_dim = 6
        
        self.emb_prot = nn.Linear(self.prot_raw_dim, hidden_nf)
        self.emb_mol = nn.Linear(self.mol_raw_dim, hidden_nf)
        self.emb_sas = nn.Linear(self.sas_raw_dim, hidden_nf)
        
        # Input to SumOuterModel: [H] + [ProtMask(1)] + [MolMask(1)] = H + 2
        self.in_node_nf = hidden_nf + 2
        self.out_node_dim = 1 # Not used for sequence output
        
        if isinstance(activation, str):
            activation = get_activation(activation)

        self.model = SumOuterModel(
            hidden_dim=hidden_nf,
            n_layers=n_layers,
            in_node_dim=self.in_node_nf,
            out_node_dim=self.out_node_dim # Dummy
        )
        
        # Head for SAS scoring: Input is Outer Product (H * H)
        self.sas_head = nn.Linear(hidden_nf * hidden_nf, 1)

    def forward(self, batch, training=False):
        # Unpack batch
        # prot: [B, Np, Fp]
        # mol: [B, Nm, Fm]
        # sas: [B, Ns, Fs]
        
        prot = batch['prot']
        mol = batch['mol']
        sas = batch['sas']
        
        # Masks [B, N_pm]
        prot_mask = batch['prot_mask'] 
        mol_mask = batch['mol_mask']
        
        # Labels [B, N_s]
        labels_pocket = batch['is_pocket']
        labels_decoy = batch['is_decoy']
        
        B = prot.shape[0]
        
        # 1. Embed Modalities
        h_prot = self.emb_prot(prot) # [B, Np, H]
        h_mol = self.emb_mol(mol)    # [B, Nm, H]
        h_sas = self.emb_sas(sas)    # [B, Ns, H]
        
        # 2. Concatenate Prot and Mol to form Context Source
        # [B, Np+Nm, H]
        h_pm = torch.cat([h_prot, h_mol], dim=1) 
        
        # 3. Concat Masks to h_pm
        # Masks are [B, N_pm]. Stack to [B, N_pm, 2]
        masks_feat = torch.stack([
            prot_mask.float(),
            mol_mask.float()
        ], dim=-1)
        
        h_pm_in = torch.cat([h_pm, masks_feat], dim=-1) # [B, N_pm, H+2]
        
        # 4. Run SumOuterModel on Prot+Mol
        # We only care about the global_context returned
        # output sequence is ignored
        _, global_context = self.model(h_pm_in) # global_context: [B, H]
        
        # 5. SAS Interaction (Outer Product)
        # h_sas: [B, Ns, H]
        # context: [B, H]
        # Outer Product: [B, Ns, H, H]
        # einsum 'bnh, bc -> bnhc'
        outer = torch.einsum('bnh, bc -> bnhc', h_sas, global_context)
        
        # Flatten feature dim: [B, Ns, H*H]
        outer_flat = outer.reshape(B, -1, self.hidden_nf * self.hidden_nf)
        
        # 6. Predict Score
        pred_scores = self.sas_head(outer_flat).squeeze(-1) # [B, Ns]
        
        # 7. Compute Loss
        score_pocket = torch.sum(pred_scores * labels_pocket, dim=1) # [B]
        score_decoy = torch.sum(pred_scores * labels_decoy, dim=1) # [B]
        
        margin = 1.0
        losses = F.relu(score_decoy - score_pocket + margin)
        loss = losses.mean()
        
        metrics = {}
        metrics['loss'] = loss
        metrics['score_pos'] = score_pocket.mean()
        metrics['score_neg'] = score_decoy.mean()
        metrics['accuracy'] = (score_pocket > score_decoy).float().mean()
        
        return metrics

    def sample_chain(self, batch, **kwargs):
        # Inference Logic (same as forward up to prediction)
        prot = batch['prot']
        mol = batch['mol']
        sas = batch['sas']
        prot_mask = batch['prot_mask'] 
        mol_mask = batch['mol_mask']
        
        B = prot.shape[0]
        
        # 1. Embed
        h_prot = self.emb_prot(prot) # [B, Np, H]
        h_mol = self.emb_mol(mol)    # [B, Nm, H]
        h_sas = self.emb_sas(sas)    # [B, Ns, H]
        
        # 2. Concat Prot+Mol
        h_pm = torch.cat([h_prot, h_mol], dim=1) 
        masks_feat = torch.stack([prot_mask.float(), mol_mask.float()], dim=-1)
        h_pm_in = torch.cat([h_pm, masks_feat], dim=-1)
        
        # 3. Model
        _, global_context = self.model(h_pm_in)
        
        # 4. SAS Interaction & Prediction
        # h_sas: [B, Ns, H]
        # context: [B, H]
        outer = torch.einsum('bnh, bc -> bnhc', h_sas, global_context)
        outer_flat = outer.reshape(B, -1, self.hidden_nf * self.hidden_nf)
        pred_scores = self.sas_head(outer_flat).squeeze(-1) # [B, Ns]
        
        return pred_scores
