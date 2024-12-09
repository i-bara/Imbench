from .nets.sha import GCNConv, GATConv, SAGEConv
from .gnnv3 import gnn, GnnModel, GNN
import torch
from torch import nn, optim
from torch.nn import functional as F
from gens import MeanAggregation_ens
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch, dropout_adj, to_dense_adj
from torch_sparse import SparseTensor
import numpy as np
import random
from typing import Tuple
from torch_geometric.utils import to_undirected


class Bat(gnn):
    def parse_args(parser):
        parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')


    def __init__(self, args):
        super().__init__(args)
        self.use(GnnModel)
        self.y_pred = None
        self.mode = "bat1"
        
        
    def get_group_mean(
        self,
        values: torch.Tensor, 
        labels: torch.Tensor, 
        classes: torch.Tensor,
    ):
        """
        Computes the mean of values within each class.

        Parameters:
        - values: torch.Tensor
            Values to compute the mean of.
        - labels: torch.Tensor
            Labels corresponding to values.
        - classes: torch.Tensor
            Classes for which to compute the mean.

        Returns:
        - new_values: torch.Tensor
            Mean values for each class.
        """
        new_values = torch.zeros_like(values)
        for i in classes:
            mask = labels == i
            new_values[mask] = values[mask].mean()
        return new_values


    def get_node_risk(self, y_pred_proba: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes node risk based on predicted probabilities.

        Parameters:
        - y_pred_proba: torch.Tensor
            Predicted class probabilities.
        - y_pred: torch.Tensor
            Predicted labels.

        Returns:
        - node_risk: torch.Tensor
            Node risk scores.
        """
        # compute node pred
        node_unc = 1 - y_pred_proba.max(axis=1).values
        # compute class-aware relative pred
        node_unc_class_mean = self.get_group_mean(node_unc, y_pred, self.classes)
        node_risk = (node_unc - node_unc_class_mean).clip(min=0)
        # calibrate node risk w.r.t class weights
        node_risk *= self.train_class_weights[y_pred]
        return node_risk
    
    
    def get_connectivity_distribution_sparse(
        self,
        y_pred: torch.Tensor,
        edge_index: torch.Tensor,
        n_class: int,
        n_node: int,
        n_edge: int,
    ):
        """
        Computes the distribution of connectivity labels.

        Parameters:
        - y_pred: torch.Tensor
            Predicted labels.
        - edge_index: torch.Tensor
            Edge indices (sparse).
        - n_class: int
            Number of classes.
        - n_node: int
            Number of nodes.
        - n_edge: int
            Number of edges.

        Returns:
        - neighbor_y_distr: torch.Tensor
            Normalized connectivity label distribution.
        """

        device = y_pred.device
        edge_dest_class = torch.zeros(
            (n_edge, n_class), dtype=torch.int, device=device
        ).scatter_(
            1, y_pred[edge_index[1]].unsqueeze(1), 1
        )  # [n_edges, n_class]
        neighbor_y_distr = (
            torch.zeros((n_node, n_class), dtype=torch.int, device=device)
            .scatter_add_(
                dim=0,
                index=edge_index[0].repeat(n_class, 1).T,
                src=edge_dest_class,
            )
            .float()
        )  # [n_nodes, n_class]

        # row-wise normalization
        neighbor_y_distr /= neighbor_y_distr.sum(axis=1).reshape(-1, 1)
        neighbor_y_distr = neighbor_y_distr.nan_to_num(0)

        return neighbor_y_distr


    def estimate_node_posterior_likelihood(
        self, y_pred_proba: torch.Tensor, y_neighbor_distr: torch.Tensor
    ):
        """
        Estimates node posterior likelihood for each class.

        Parameters:
        - y_pred_proba: torch.Tensor
            Predicted class probabilities.
        - y_neighbor_distr: torch.Tensor
            Connectivity label distribution.

        Returns:
        - node_posterior: torch.Tensor
            Node posterior likelihood.
        """
        mode = self.mode
        if mode == "bat0":
            node_posterior = y_pred_proba
        elif mode == "bat1":
            node_posterior = y_neighbor_distr
        else:
            raise NotImplementedError
        return node_posterior


    def get_virual_link_proba(self, node_posterior: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes virtual link probabilities based on node posterior likelihood.

        Parameters:
        - node_posterior: torch.Tensor
            Node posterior likelihood.
        - y_pred: torch.Tensor
            Predicted labels.

        Returns:
        - virtual_link_proba: torch.Tensor
            Virtual link probabilities.
        """
        # set likelihood to current predicted class as 0
        node_posterior *= 1 - F.one_hot(y_pred, num_classes=self.n_cls)
        node_posterior = node_posterior.clip(min=0)
        # row-wise renormalize
        node_posterior /= node_posterior.sum(axis=1).reshape(-1, 1)
        virtual_link_proba = node_posterior.nan_to_num(0)
        return virtual_link_proba
    
    
    def edge_sampling(
        self,
        edge_index: torch.Tensor,
        edge_sampling_proba: torch.Tensor,
    ):
        """
        Performs edge sampling based on probability.

        Parameters:
        - edge_index: torch.Tensor
            Edge indices.
        - edge_sampling_proba: torch.Tensor
            Edge sampling probabilities.

        Returns:
        - sampled_edge_index: torch.Tensor
            Sampled edge indices.
        """
        assert edge_sampling_proba.min() >= 0 and edge_sampling_proba.max() <= 1
        edge_sample_mask = torch.rand_like(edge_sampling_proba) < edge_sampling_proba
        return edge_index[:, edge_sample_mask]


    def get_virtual_node_features(
        self, 
        x: torch.Tensor, 
        y_pred: torch.Tensor, 
        classes: list, 
        ):
        """
        Computes virtual node features based on predicted labels.

        Parameters:
        - x: torch.Tensor
            Node features.
        - y_pred: torch.Tensor
            Predicted labels.
        - classes: list
            Unique classes in the dataset.

        Returns:
        - virtual_node_features: torch.Tensor
            Virtual node features for each class.
        """
        return torch.stack([x[y_pred == label].mean(axis=0) for label in classes])


    def augment(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        y_pred_proba = self.y_pred
        y_pred = y_pred_proba.argmax(axis=1)
        y_label = y.argmax(axis=1)
        y_pred[self.mask('train')] = y_label[self.mask('train')]
        
        classes, train_class_counts = y_label.unique(return_counts=True)
        self.classes = classes
        self.train_class_weights = train_class_counts / train_class_counts.max()

        # compute node_risk and virtual link probability
        node_risk = self.get_node_risk(y_pred_proba, y_pred)

        if self.mode == "bat0":
            y_neighbor_distr = None
        else:
            y_neighbor_distr = self.get_connectivity_distribution_sparse(
                y_pred, edge_index, self.n_cls, self.n_node, self.n_edge
            )

        node_posterior = self.estimate_node_posterior_likelihood(
            y_pred_proba, y_neighbor_distr
        )
        virtual_link_proba = self.get_virual_link_proba(node_posterior, y_pred)

        # assign link probability w.r.t node risk
        virtual_link_proba *= node_risk.reshape(-1, 1)

        # sample virtual edge_index w.r.t given probability
        virtual_adj = virtual_link_proba.T.to_sparse().coalesce()
        edge_index_candidates, edge_sampling_proba = (
            virtual_adj.indices(),
            virtual_adj.values(),
        )

        new_edge_index = self.edge_sampling(
            edge_index_candidates, edge_sampling_proba
        )
        new_edge_index[
            0
        ] += self.n_node  # adjust index to match original node index
        
        new_edge_index = to_undirected(new_edge_index)

        # compute virtual node features
        new_y = y_label[self.mask('train')].unique()
        new_x = self.get_virtual_node_features(x, y_pred, new_y)
        new_y = F.one_hot(new_y, num_classes=self.n_cls).to(torch.float)

        return new_x, new_edge_index, new_y


    def new(self, old):
        x, edge_index, y = old
        x = torch.zeros((0, x.shape[1]), dtype=x.dtype, device=x.device)
        edge_index = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        if len(y.shape) == 1:  # hard label
            y = torch.zeros((0,), dtype=y.dtype, device=y.device)
        else:  # soft label
            y = torch.zeros((0, y.shape[1]), dtype=y.dtype, device=y.device)
        return x, edge_index, y


    def cat(self, old, *args):
        x, edge_index, y = old
        for new in args:
            new_x, new_edge_index, new_y = new
            x = torch.cat((x, new_x), dim=0)
            edge_index = torch.cat((edge_index, new_edge_index), dim=1)
            y = torch.cat((y, new_y), dim=0)
        return x, edge_index, y


    def epoch_output(self, epoch):
        if self.training:
            x, edge_index, y = self.data.x, self.data.edge_index, torch.nn.functional.one_hot(self.data.y, num_classes=self.n_cls).to(torch.float)
            # x_smote, edge_index_smote, y_smote = self.mixup_smote_new(x, edge_index, y, epoch=epoch, sampling_list=1.0, connect=True, score=self.score)
            # x_sha, edge_index_sha, y_sha = self.mixup_sha_new(x, edge_index, y, epoch=epoch, sampling_list=1.0, score=self.score, n_node=x.shape[0] + x_smote.shape[0])
            
            if epoch > self.args.warmup:
                x_sha, edge_index_sha, y_sha = self.augment(x, edge_index, y)
                # x_sha, edge_index_sha, y_sha = self.igraphmix_ens(x, edge_index, y, self.lam, self.src, self.dst, n_node=x.shape[0], connect=True, saliency=self.saliency, disk_kl=None)
                self.sha = slice(x.shape[0], x.shape[0] + x_sha.shape[0])

                x, edge_index, y = self.cat(
                    (x, edge_index, y),
                    # (x_smote, edge_index_smote, y_smote),
                    # (x_sha, edge_index_sha, y_sha),
                    (x_sha, edge_index_sha, y_sha)
                    )

            self.data.x = x
            self.data.edge_index = edge_index
            self.data.y = y
            n_sample = x.shape[0]
            for name, mask in self.masks.items():
                if name == 'train':
                    self.masks[name] = torch.cat((mask, torch.ones(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)
                else:
                    self.masks[name] = torch.cat((mask, torch.zeros(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)
        else:
            x, edge_index, y = self.data.x, self.data.edge_index, torch.nn.functional.one_hot(self.data.y, num_classes=self.n_cls).to(torch.float)
            # x, edge_index, y = self.data.x, self.data.edge_index, self.data.y

        # Get grad of x as saliency
        self.data.x.requires_grad = True
        
        logits = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y)
        self.y_pred = torch.softmax(logits, dim=1).detach()
        
        return logits
    
    
    def logits(self, output):
        y_pred = torch.softmax(output, dim=1)
        return y_pred


    def loss(self, output, y):
        y_pred = torch.softmax(output, dim=1)
        return F.cross_entropy(y_pred, y)
