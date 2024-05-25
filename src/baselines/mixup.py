from .nets import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModelWithEncoder
import torch
from torch import nn, optim
from torch.nn import functional as F
from .baseline import Baseline, Timer
from scipy.spatial.distance import pdist, squareform
import random
import tqdm
import numpy as np
from renode import index2dense


class GNN(nn.Module):
    ''' 
    A GNN backbone, such as GCN, GAT and SAGE.
    '''
    def __init__(self, Conv, n_feat, n_hid, n_cls, dropout, n_layer, **kwargs):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([Conv(n_feat if layer == 0 else n_hid, 
                                         n_cls if layer + 1 == n_layer else n_hid, **kwargs) 
                                         for layer in range(n_layer)])

        self.dropout = dropout
        # self.x_dropout = x_dropout
        # self.edge_index_dropout = edge_index_dropout

        self.reg_params = self.convs[:-1].parameters()
        self.non_reg_params = self.convs[-1].parameters()


    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # x = F.dropout(x, p=self.x_dropout, training=self.training)
        # edge_index = F.dropout(edge_index, p=self.edge_index_dropout, training=self.training)

        for conv in self.convs[:-1]:
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)
        # , edge_index
        return x


class GnnModel(nn.Module):
    def __init__(self, args, baseline):
        super(GnnModel, self).__init__()
        self.args = args
        self.baseline = baseline
        self.criterion_default = self.criterion
        self.criterion_dict = dict()

        self.regularizations = set()

        self.conv_dict, self.gnn_kwargs = self.config_gnn()

        self.classifier = GNN(Conv=self.conv_dict[args.net], 
                              n_feat=baseline.n_feat, n_hid=args.feat_dim, n_cls=baseline.n_cls, 
                              dropout=args.dropout, 
                            #   x_dropout=args.x_dropout, 
                            #   edge_index_dropout=args.edge_index_dropout, 
                              n_layer=args.n_layer, **self.gnn_kwargs)


    def config_gnn(self):
        '''
        Return:
        conv_dict: mapping the conv name to the conv layer.
        gnn_kwargs: kwargs used to create GNN backbone.
        '''
        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
        }
        gnn_kwargs = dict()
        return conv_dict, gnn_kwargs
    

    @DeprecationWarning
    def regularization(func):
        def wrapper(self, *args, **kwargs):
            self.regularizations.add(func)
            return func(self, *args, **kwargs)
        return wrapper


    def criterion(self, output, y, mask, weight):
        return F.cross_entropy(output[mask], y[mask], weight=weight)


    def forward(self, x, edge_index, y=None, mask=None, weight=None, logit=False, phase=None, reg=None, **kwargs):
        if mask is None:
            mask = self.baseline.mask()
        
        output = self.classifier(x=x, edge_index=edge_index, **kwargs)
        if logit:
            return output
        
        if phase is None:
            criterions = self.criterion_default
        else:
            criterions = self.criterion_dict[phase]

        if type(criterions) == dict:
            loss = 0
            for criterion, criterion_weight in criterions.items():
                if criterion in self.regularizations:
                    loss += criterion_weight * criterion(**reg)
                else:
                    loss += criterion_weight * criterion(output=output, y=y, mask=mask, weight=weight)
        elif type(criterions) == list:
            loss = 0
            for criterion in criterions:
                if criterion in self.regularizations:
                    loss += criterion(**reg)
                else:
                    loss += criterion(output=output, y=y, mask=mask, weight=weight)
        else:
            criterion = criterions
            if criterion in self.regularizations:
                loss = criterion(**reg)
            else:
                loss = criterion(output=output, y=y, mask=mask, weight=weight)
        
        return loss


class mixup(gnn):
    def parse_args(parser):
        # up_scale=1.0, im_class_num=3, scale=0.0
        Baseline.add_argument(parser, "--up_scale", type=float, default=1, help="")
        Baseline.add_argument(parser, "--scale", type=float, default=0, help="")
        Baseline.add_argument(parser, "--im_class_ratio", type=int, default=0.33, help="")


    def __init__(self, args):
        super().__init__(args)
        self.use(GnnModelWithEncoder)
        self.data_original = self.data.clone().detach()
        self.masks_original = dict()
        for name, mask in self.masks.items():
            self.masks_original[name] = mask.clone().detach()


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
            
            if self.args.up_scale == 0:
                c_up_scale = int(avg_number / idx_num + self.args.scale) - 1
                if c_up_scale >= 0:
                    up_scale_rest = avg_number / idx_num + self.args.scale - 1 - c_up_scale
                else:
                    c_up_scale = 0
                    up_scale_rest = 0
                # print(round(scale, 2), round(c_up_scale, 2), round(up_scale_rest, 2))
            else:
                c_up_scale = int(self.args.up_scale)
                up_scale_rest = self.args.up_scale - c_up_scale

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


    def epoch_loss(self, epoch, mode='test'):
        embed = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data_original.y, logit=True, phase='embed', **self.forward_kwargs)

        self.data = self.data_original.clone().detach()
        self.masks = dict()
        for name, mask in self.masks_original.items():
            self.masks[name] = mask.clone().detach()

        # embed, edge_index, y = embed, self.data.edge_index, self.data_original.y
        embed, edge_index, y = self.mixup(embed, self.data.edge_index, self.data_original.y)

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
