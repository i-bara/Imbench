from .nets import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModelWithEncoder
import torch
from torch import nn
from torch.nn import functional as F
from .baseline import Baseline
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np


class mixup(gnn):
    def parse_args(parser):
        # up_scale=1.0, im_class_num=3, scale=0.0
        Baseline.add_argument(parser, "--up_scale", type=float, default=1, help="")
        Baseline.add_argument(parser, "--scale", type=float, default=0, help="")
        Baseline.add_argument(parser, "--im_class_ratio", type=float, default=0.33, help="")


    def __init__(self, args):
        super().__init__(args)
        self.use(GnnModelWithEncoder)
        self.data_original = self.data.clone().detach()
        self.masks_original = dict()
        for name, mask in self.masks.items():
            self.masks_original[name] = mask.clone().detach()


    def scale(self, avg_number, idx_num):
        if self.args.up_scale == 0:
            up_scale = int(avg_number / idx_num + self.args.scale) - 1
            if up_scale >= 0:
                up_scale_rest = avg_number / idx_num + self.args.scale - 1 - up_scale
            else:
                up_scale = 0
                up_scale_rest = 0
            # print(round(scale, 2), round(c_up_scale, 2), round(up_scale_rest, 2))
        else:
            up_scale = int(self.args.up_scale)
            up_scale_rest = self.args.up_scale - up_scale

        return up_scale, up_scale_rest


    def mixup(self, x, edge_index, y):
        im_class_num = int(self.args.im_class_ratio * self.n_cls)

        c_largest = self.n_cls - 1
        avg_number = int(self.num('train') / (self.n_cls))

        idx_train = self.idx('train')

        for i in range(im_class_num):
            c = c_largest - i
            # mask = self.mask('train') & self.mask(c)
            # idx = self.idx(mask=mask)
            idx = idx_train[(y == c)[idx_train]]
            idx_num = idx.shape[0]
            
            c_up_scale, up_scale_rest = self.scale(avg_number=avg_number, idx_num=idx_num)
            
            for j in range(c_up_scale):

                chosen_embed = x[idx, :]
                distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
                np.fill_diagonal(distance, distance.max() + 100)
                idx_neighbor = distance.argmin(axis=-1)
                
                interp_place = random.random()

                x_new = x[idx, :] + (x[idx_neighbor, :] - x[idx, :]) * interp_place
                y_new = y.new(torch.Size((idx_num, 1))).reshape(-1).fill_(c_largest - i)
                idx_new = idx_train.new(np.arange(x.shape[0], x.shape[0] + idx_num))

                x = torch.cat((x, x_new), 0)
                y = torch.cat((y, y_new), 0)
                idx_train = torch.cat((idx_train, idx_new), 0)

            if up_scale_rest != 0.0:

                num = int(idx_num * up_scale_rest)
                idx = idx[:num]

                chosen_embed = x[idx, :]
                distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
                np.fill_diagonal(distance, distance.max() + 100)
                idx_neighbor = distance.argmin(axis=-1)
                
                interp_place = random.random()

                x_new = x[idx, :] + (x[idx_neighbor, :] - x[idx, :]) * interp_place
                y_new = y.new(torch.Size((idx.shape[0], 1))).reshape(-1).fill_(c_largest - i)
                idx_new = idx_train.new(np.arange(x.shape[0], x.shape[0] + idx_num))

                x = torch.cat((x, x_new), 0)
                y = torch.cat((y, y_new), 0)
                idx_train = torch.cat((idx_train, idx_new), 0)

        return x, edge_index, y


    def load(self):
        self.data = self.data_original.clone().detach()
        self.masks = dict()
        for name, mask in self.masks_original.items():
            self.masks[name] = mask.clone().detach()

    
    def store(self, y):
        self.data.y = y
        n_sample = y.shape[0]
        for name, mask in self.masks_original.items():
            if name == 'train':
                self.masks[name] = torch.cat((mask, torch.ones(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)
            else:
                self.masks[name] = torch.cat((mask, torch.zeros(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)


    def epoch_loss(self, epoch, mode='test'):
        embed = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data_original.y, logit=True, phase='embed', **self.forward_kwargs)

        self.load()

        # embed, edge_index, y = embed, self.data.edge_index, self.data_original.y
        embed, edge_index, y = self.mixup(embed, self.data.edge_index, self.data_original.y)

        self.store(y)

        if mode == 'test':
            return self.model(x=embed, edge_index=edge_index, y=y, logit=True, **self.forward_kwargs)
        else:
            return self.model(x=embed, edge_index=edge_index, y=y, mask=self.mask(mode), **self.forward_kwargs)
