from .nets import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModelWithEncoder
from .mixup import mixup
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils.dropout import dropout_adj
from .baseline import Baseline
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np


class upsample(mixup):
    def parse_args(parser):
        # up_scale=1.0, im_class_num=3, scale=0.0
        Baseline.add_argument(parser, "--mixup_alpha", type=float, default=100, help="")


    def mixup(self, x, edge_index, y):
        x, edge_index, y = x.clone(), edge_index.clone(), y.clone()
        avg_num = int(self.num('train') / (self.n_cls))
        x_new = torch.zeros((0, x.shape[1]), dtype=x.dtype, device=x.device)
        edge_index_new = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        y_new = torch.zeros((0,), dtype=y.dtype, device=y.device)
        num_now = self.num()
        for c in range(self.n_cls):
            c_idx = self.idx(mask=self.mask('train') & self.mask(c))
            c_num = c_idx.shape[0]
            if c_num < avg_num:
                c_new_idx = c_idx[torch.multinomial(torch.ones(c_num, dtype=torch.float, device=self.device), avg_num - c_num, replacement=True)]
                x_new = torch.cat((x_new, x[c_new_idx]), dim=0)
                y_new = torch.cat((y_new, y[c_new_idx]), dim=0)
                for i in range(avg_num - c_num):
                    n = c_new_idx[i].item()
                    edge_index_new_ = edge_index[:, (edge_index[0] == n) | (edge_index[1] == n)]
                    edge_index_new_[edge_index_new_ == n] = num_now
                    edge_index_new = torch.cat((edge_index_new, edge_index_new_), dim=1)
                    num_now += 1

        x = torch.cat((x, x_new), dim=0)
        edge_index = torch.cat((edge_index, edge_index_new), dim=1)
        y = torch.cat((y, y_new), dim=0)
        return x, edge_index, y


    def epoch_loss(self, epoch, mode='test'):
        print(mode)
        embed = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data_original.y, logit=True, phase='embed', **self.forward_kwargs)

        self.data = self.data_original.clone().detach()
        self.masks = dict()
        for name, mask in self.masks_original.items():
            self.masks[name] = mask.clone().detach()

        # embed, edge_index, y = embed, self.data.edge_index, self.data_original.y
        if mode == 'train':
            print(embed.shape, self.data.edge_index.shape, self.data_original.y.shape)
            embed, edge_index, y = self.mixup(embed, self.data.edge_index, self.data_original.y)
            print(embed.shape, edge_index.shape, y.shape)
        else:
            embed, edge_index, y = embed, self.data.edge_index, self.data_original.y

        self.data.y = y
        n_sample = embed.shape[0]
        for name, mask in self.masks_original.items():
            if name == 'train':
                self.masks[name] = torch.cat((mask, torch.ones(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)
            else:
                self.masks[name] = torch.cat((mask, torch.zeros(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)

        if mode == 'test':
            return self.model(x=embed, edge_index=edge_index, y=y, logit=True, **self.forward_kwargs)
        else:
            return self.model(x=embed, edge_index=edge_index, y=y, mask=self.mask(mode), **self.forward_kwargs)
