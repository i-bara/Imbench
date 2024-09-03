from .nets import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModel, GNN
import torch
from torch import nn
from torch.nn import functional as F
from .baseline import Baseline
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np


class DpgnnModel(GnnModel):
    def __init__(self, args, baseline):
        super().__init__(args, baseline)
        self.encoder = GNN(Conv=self.conv_dict[args.net], 
                              n_feat=baseline.n_feat, n_hid=args.feat_dim, n_cls=args.feat_dim, 
                              dropout=args.dropout, 
                              n_layer=args.n_layer, **self.gnn_kwargs)
        self.dist_encoder = DistEncoder(n_hid=args.feat_dim, n_cls=baseline.n_cls)
        
        self.reg_params = list(self.encoder.reg_params) + list(self.dist_encoder.reg_params)
        self.non_reg_params = list(self.encoder.non_reg_params) + list(self.dist_encoder.non_reg_params)


    def criterion(self, output, y, mask, weight):
        return F.cross_entropy(output, y, weight=weight)


    def _output(self, x, edge_index, phase, **kwargs):
        support, query = self.baseline.episodic_generator()

        embed = self.encoder(x=x, edge_index=edge_index)

        support_embed = [embed[support[i]] for i in range(self.baseline.n_cls)]

        query_embed = [embed[query[i]] for i in range(self.baseline.n_cls)]
        query_embed = torch.stack(query_embed, dim=0)

        proto_embed = [torch.mean(support_embed[i], dim=0) for i in range(self.baseline.n_cls)]
        proto_embed = torch.stack(proto_embed, dim=0)

        query_dist_embed = self.dist_encoder(query_embed, proto_embed)
        proto_dist_embed = self.dist_encoder(proto_embed, proto_embed)

        return torch.mm(query_dist_embed, proto_dist_embed)
    

    def forward(self, x, edge_index, y=None, mask=None, weight=None, logit=False, phase=None, reg=None, **kwargs):
        if mask is None:
            mask = self.baseline.mask()

        embed = self.encoder(x=x, edge_index=edge_index)
        if logit:
            support, query = self.baseline.episodic_generator()

            support_embed = [embed[support[i]] for i in range(self.baseline.n_cls)]

            query_embed = embed[mask]

            proto_embed = [torch.mean(support_embed[i], dim=0) for i in range(self.baseline.n_cls)]
            proto_embed = torch.stack(proto_embed, dim=0)

            query_dist_embed = self.dist_encoder(query_embed, proto_embed)
            proto_dist_embed = self.dist_encoder(proto_embed, proto_embed)

            return torch.mm(query_dist_embed, proto_dist_embed)
        
        support, query = self.baseline.episodic_generator()

        embed = self.encoder(x=x, edge_index=edge_index)

        support_embed = [embed[support[i]] for i in range(self.baseline.n_cls)]

        query_embed = [embed[query[i]] for i in range(self.baseline.n_cls)]
        query_embed = torch.stack(query_embed, dim=0)

        proto_embed = [torch.mean(support_embed[i], dim=0) for i in range(self.baseline.n_cls)]
        proto_embed = torch.stack(proto_embed, dim=0)

        query_dist_embed = self.dist_encoder(query_embed, proto_embed)
        proto_dist_embed = self.dist_encoder(proto_embed, proto_embed)

        return F.cross_entropy(torch.mm(query_dist_embed, proto_dist_embed), y, weight=weight)


class dpgnn(gnn):
    def parse_args(parser):
        Baseline.add_argument(parser, "--episodic_samp", type=float, default=1, help="")


    def __init__(self, args):
        super().__init__(args)
        self.use(DpgnnModel)

    
    def episodic_generator(self):
        support = []
        query = []
        for c in range(self.n_cls):
            mask = self.mask('train') & self.mask(c)
            idx = self.idx(mask=mask)
            num = self.num(mask=mask)
            # print(idx)
            # print(idx.shape)
            # print(num)
            # exit()

            sample_idx = idx[torch.randperm(idx.size(0))[:int(num * self.args.episodic_samp)]]

            # sample_idx = random.sample(idx, int(num * self.args.episodic_samp))

            if(len(sample_idx) >= 2):
                support_idx = sample_idx[1:]
                query_idx = sample_idx[0]
            else:
                support_idx = sample_idx[:]
                query_idx = sample_idx[0]

            support.append(support_idx)
            query.append(query_idx)

        return support, query


    def epoch_loss(self, epoch, mode='test'):
        # support, query = self.episodic_generator()

        # embed = self.model.encoder(x=self.data.x, edge_index=self.data.edge_index)

        # support_embed = [embed[support[i]] for i in range(self.n_cls)]

        # query_embed = [embed[query[i]] for i in range(self.n_cls)]
        # query_embed = torch.stack(query_embed, dim=0)

        # proto_embed = [torch.mean(support_embed[i], dim=0) for i in range(self.n_cls)]
        # proto_embed = torch.stack(proto_embed, dim=0)

        # query_dist_embed = self.model.dist_encoder(query_embed, proto_embed)
        # proto_dist_embed = self.model.dist_encoder(proto_embed, proto_embed)

        # output = torch.mm(query_dist_embed, proto_dist_embed)

        # loss1 = F.cross_entropy(output, torch.arange(self.n_cls, dtype=self.data.y.dtype, device=self.device))

        if mode == 'test':
            return self.model(x=self.data.x, edge_index=self.data.edge_index, y=torch.arange(self.n_cls, dtype=self.data.y.dtype, device=self.device), logit=True, **self.forward_kwargs)
        else:
            return self.model(x=self.data.x, edge_index=self.data.edge_index, y=torch.arange(self.n_cls, dtype=self.data.y.dtype, device=self.device), mask=self.mask(mode), **self.forward_kwargs)


class prototype(torch.nn.Module):
    def __init__(self):
        super(prototype, self).__init__()


    def forward(self, x):
        return torch.mean(x, dim=0)


class DistEncoder(torch.nn.Module):
    def __init__(self, n_hid, n_cls):
        super(DistEncoder, self).__init__()
        self.lin = nn.Linear(n_hid * n_cls, n_cls)

        self.reg_params = list()
        self.non_reg_params = list(self.lin.parameters())


    def forward(self, query, proto):
        d1 = query.size(0)
        d2 = proto.size(0)

        query = torch.repeat_interleave(query, d2, dim=0)
        proto = torch.tile(proto, (d1, 1))

        dist = self.lin((query - proto).view(d1, -1))

        return dist
