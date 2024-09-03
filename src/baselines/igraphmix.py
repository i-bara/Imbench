from .nets import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModelWithEncoder
from .upsample import upsample
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils.dropout import dropout_adj
from .baseline import Baseline
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np


class igraphmix(upsample):
    def parse_args(parser):
        # up_scale=1.0, im_class_num=3, scale=0.0
        Baseline.add_argument(parser, "--mixup_alpha", type=float, default=100, help="")
        Baseline.add_argument(parser, "--upsample", default="mean", type=str,
                            choices=["none", "mean"])


    def mixup(self, x, edge_index, y):
        if self.args.upsample == 'mean':
            x, edge_index, y = super().mixup(x, edge_index, y)
            self.store(y)

        # 0) Sample lambda
        lambda_ = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)

        # 1) Index Permutation
        idx = torch.arange(x.shape[0], device=self.device)
        idx[self.idx('train')] = self.idx('train')[torch.randperm(self.num('train'))]
        self.debug(idx)

        # index = torch.cat((self.idx('train'), self.idx(mask=torch.logical_not(self.mask('train')))), dim=0)
        # index_new = torch.zeros(index.shape[0], dtype=torch.long)
        # index_new[index] = torch.arange(0, index.shape[0])

        y = F.one_hot(y, num_classes=self.n_cls)

        # 2) Node feature Mixup
        x = lambda_ * x + (1-lambda_) * x[idx]

        # 3) Label Mixup
        y = lambda_ * y + (1-lambda_) * y[idx]

        # # # 4) Edge Mixup
        # edge_index = data.edge_index.clone().detach()
        # row, col = edge_index.clone().detach()[0], edge_index.clone().detach()[1]
        # row, col = index_new.clone().detach()[row], index_new.clone().detach()[col]
        # edge_index_perm = torch.stack([row, col], dim=0)

        # # 5) Prepare Mixup Graph
        # edge_index, _ = dropout_adj(edge_index, p=1-lambda_, training=True)
        # edge_index_perm, _ = dropout_adj(edge_index_perm, p=lambda_, training=True)
        # edge_index_mixup = torch.cat((edge_index, edge_index_perm), dim=1)

        edge_index = torch.cat((
            dropout_adj(edge_index, p=1-lambda_, training=True)[0], 
            dropout_adj(self.inv2(idx)[edge_index], p=lambda_, training=True)[0]), dim=1)

        # data_mixup = Data(x=x_mixup, y=y_mixup, edge_index=edge_index_mixup)
        # data_mixup.train_mask = data.train_mask
        # data_mixup.val_mask = data.val_mask
        # data_mixup.test_mask = data.test_mask

        # im_class_num = int(self.args.im_class_ratio * self.n_cls)

        # c_largest = self.n_cls - 1
        # avg_number = int(self.num('train') / (self.n_cls))

        # idx_train = self.idx('train')

        # for i in range(self.n_cls):
        #     for j in range(self.n_cls):
        #         if i == j:
        #             break

        #         i_idx = idx_train[(y == i)[idx_train]]
        #         i_idx_num = idx.shape[0]

        #         j_idx = idx_train[(y == j)[idx_train]]
        #         j_idx_num = idx.shape[0]
                
        #         i_up_scale, i_up_scale_rest = self.scale(avg_number=avg_number, idx_num=i_idx_num)
        #         j_up_scale, j_up_scale_rest = self.scale(avg_number=avg_number, idx_num=j_idx_num)

        #         for ki in range(i_up_scale):

        #             chosen_embed = x[idx, :]
        #             distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
        #             np.fill_diagonal(distance, distance.max() + 100)
        #             idx_neighbor = distance.argmin(axis=-1)
                    
        #             interp_place = random.random()

        #             x_new = x[idx, :] + (x[idx_neighbor, :] - x[idx, :]) * interp_place
        #             y_new = y.new(torch.Size((idx_num, 1))).reshape(-1).fill_(c_largest - i)
        #             idx_new = idx_train.new(np.arange(x.shape[0], x.shape[0] + idx_num))

        #             x = torch.cat((x, x_new), 0)
        #             y = torch.cat((y, y_new), 0)
        #             idx_train = torch.cat((idx_train, idx_new), 0)

        #         if up_scale_rest != 0.0:

        #             num = int(idx_num * up_scale_rest)
        #             idx = idx[:num]

        #             chosen_embed = x[idx, :]
        #             distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
        #             np.fill_diagonal(distance, distance.max() + 100)
        #             idx_neighbor = distance.argmin(axis=-1)
                    
        #             interp_place = random.random()

        #             x_new = x[idx, :] + (x[idx_neighbor, :] - x[idx, :]) * interp_place
        #             y_new = y.new(torch.Size((idx.shape[0], 1))).reshape(-1).fill_(c_largest - i)
        #             idx_new = idx_train.new(np.arange(x.shape[0], x.shape[0] + idx_num))

        #             x = torch.cat((x, x_new), 0)
        #             y = torch.cat((y, y_new), 0)
        #             idx_train = torch.cat((idx_train, idx_new), 0)

        return x, edge_index, y
