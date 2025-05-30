import pytorch_lightning as pl
import torch

from src import const
from src.gnn import GNN
from src.const import TORCH_INT
from src.datasets import (
    PocketDataset, get_dataloader, collate
)
from typing import Dict, List, Optional
from torch.nn import functional as F

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")

class YuelPocket(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        hidden_nf,
        activation='silu', n_layers=3, sin_embedding=True, normalization_factor=1, aggregation_method='sum',
        batch_size=2, lr=1e-4, torch_device='cpu', test_epochs=1, n_stability_samples=1,
        log_iterations=None, samples_dir=None, data_augmentation=False,
    ):
        super(YuelPocket, self).__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation

        self.in_node_nf = const.N_RESIDUE_TYPES + const.N_ATOM_TYPES + 1  # 氨基酸和原子的one-hot编码，+1是因为joint node
        self.in_edge_nf = 1 + 1 + 1 + 1 + const.N_RDBOND_TYPES  # distance, backbone neighbor, protein-joint, joint-mol, bond_one_hot
        self.hidden_nf = hidden_nf
        self.out_node_nf = 1  # 预测每个节点是否是口袋
        self.out_edge_nf = 1  # 预测边是否存在

        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        if type(activation) is str:
            activation = get_activation(activation)

        # in_node_nf, in_edge_nf, hidden_nf, 
        self.gnn = GNN(
            # n_layers=3, attention=False,
                 # norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2
            in_node_nf=self.in_node_nf,
            in_edge_nf=self.in_edge_nf,
            hidden_nf=hidden_nf,
            out_node_nf=self.out_node_nf,
            out_edge_nf=self.out_edge_nf,
            n_layers=n_layers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PocketDataset(
            device=self.torch_device,
            progress_bar=True,
            split='train'
        )
        self.val_dataset = PocketDataset(
            device=self.torch_device,
            progress_bar=True,
            split='val'
        )

    def train_dataloader(self, collate_fn=collate):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    def forward(self, data):

        # 'pname': sample[1],
        # 'lname': sample[2],
        # 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=self.device),
        # 'edge_index': torch.tensor(edge_index, dtype=const.TORCH_INT, device=self.device),
        # 'edge_attr': torch.tensor(edge_attr, dtype=const.TORCH_FLOAT, device=self.device),
        # 'node_mask': torch.tensor(node_mask, dtype=const.TORCH_FLOAT, device=self.device),
        # 'edge_mask': torch.tensor(edge_mask, dtype=const.TORCH_FLOAT, device=self.device),
        # 'protein_mask': torch.tensor(protein_mask, dtype=const.TORCH_FLOAT, device=self.device),
        # 'is_pocket': torch.tensor(is_pocket, dtype=const.TORCH_FLOAT, device=self.device)

        h = data['one_hot']
        edge_index = data['edge_index'].to(TORCH_INT)
        edge_attr = data['edge_attr']

        node_mask = data['node_mask']
        edge_mask = data['edge_mask']

        # feat_mask = torch.tensor(molecule_feat_mask(), device=x.device)

        pocket_pred, _ = self.gnn.forward(
            h=h,
            edge_index=edge_index,
            edge_attr=edge_attr,

            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        pocket_pred = pocket_pred * data['protein_mask']

        # sigmoid
        pocket_pred = torch.sigmoid(pocket_pred)

        return pocket_pred
    
    def loss_fn(self, pocket_pred, pocket_true, protein_mask):
        # pocket_pred: b, n, 1
        # pocket_true: b, n, 1
        # protein_mask: b, n, 1
        batch_size, n_nodes, _ = pocket_pred.shape
        
        # Reshape tensors for binary cross entropy
        pocket_pred = pocket_pred.view(batch_size * n_nodes)
        pocket_true = pocket_true.view(batch_size * n_nodes)
        protein_mask = protein_mask.view(batch_size * n_nodes)

        pocket_pred = pocket_pred * protein_mask
        
        # Apply binary cross entropy loss (without logits since input is already probability)
        loss = F.binary_cross_entropy(pocket_pred, pocket_true, reduction='none')
        
        # Apply protein mask and compute mean
        loss = (loss * protein_mask).sum() / (protein_mask.sum() + 1e-8)
        
        return loss

    def training_step(self, data, *args):
        pocket_pred = self.forward(data)
        pocket_true = data['is_pocket']
        protein_mask = data['protein_mask']

        pocket_pred = pocket_pred * protein_mask # b, n, c

        loss = self.loss_fn(pocket_pred, pocket_true, protein_mask)

        training_metrics = {
            'loss': loss,
        }
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                self.log(f'{metric_name}/train', metric, prog_bar=True)

        # self.training_step_outputs.append(training_metrics)
        return training_metrics

    def validation_step(self, data, *args):
        pocket_pred = self.forward(data)
        pocket_true = data['is_pocket']
        protein_mask = data['protein_mask']

        pocket_pred = pocket_pred * protein_mask # b, n, c

        loss = self.loss_fn(pocket_pred, pocket_true, protein_mask)

        rt = {
            'loss': loss,
        }
        self.validation_step_outputs.append(rt)
        return rt

    def test_step(self, data, *args):
        pocket_pred = self.forward(data)
        pocket_true = data['is_pocket']
        protein_mask = data['protein_mask']

        pocket_pred = pocket_pred * protein_mask # b, n, c

        loss = self.loss_fn(pocket_pred, pocket_true, protein_mask)

        rt = {
            'loss': loss,
        }
        self.test_step_outputs.append(rt)
        return rt

    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        for metric in self.test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
