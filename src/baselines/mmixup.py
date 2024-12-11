from .mix_base import mix_base
from .gnnv3 import GNN, GnnModel

import torch
from torch import nn
from torch.nn import functional as F


class MmixupGNN(GNN):
    ''' 
    A GNN backbone, such as GCN, GAT and SAGE.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_feat, n_hid, n_cls, n_layer = \
            kwargs['n_feat'], kwargs['n_hid'], kwargs['n_cls'], kwargs['n_layer']
            
        self.n_layer = n_layer
        self.lins = nn.ModuleList([nn.Linear( \
            n_feat if layer == 0 else n_hid, \
            n_cls if layer + 1 == n_layer else n_hid) \
                for layer in range(n_layer)])
    
    
    def forward(self, x
                , edge_index, edge_index_b, lam, idx, edge_weight=None, **kwargs):
        # x = F.dropout(x, p=self.x_dropout, training=self.training)
        # edge_index = F.dropout(edge_index, p=self.edge_index_dropout, training=self.training)
        xs = [x]

        if self.encoder:
            for i in range(self.n_layer):
                x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs) \
                    + self.lins[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x)
        else:
            for i in range(self.n_layer - 1):
                x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs) \
                    + self.lins[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x)
                
            x = self.convs[self.n_layer - 1](x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)

        xs_b = [x[idx] for x in xs]
        
        x = xs[0]
        x_b = xs_b[0]
        x_m = (1 - lam) * x + lam * x_b
        
        if self.encoder:
            for i in range(self.n_layer):
                x = self.convs[i](x=xs[i], edge_index=edge_index, edge_weight=edge_weight, **kwargs) \
                    + self.lins[i](x_m)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x_b = self.convs[i](x=xs_b[i], edge_index=edge_index_b, edge_weight=edge_weight, **kwargs) \
                    + self.lins[i](x_m)
                x_b = F.relu(x_b)
                x_m = (1 - lam) * x + lam * x_b
                x_m = F.dropout(x_m, p=self.dropout, training=self.training)
        else:
            for i in range(self.n_layer - 1):
                x = self.convs[i](x=xs[i], edge_index=edge_index, edge_weight=edge_weight, **kwargs) \
                    + self.lins[i](x_m)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x_b = self.convs[i](x=xs_b[i], edge_index=edge_index_b, edge_weight=edge_weight, **kwargs) \
                    + self.lins[i](x_m)
                x_b = F.relu(x_b)
                x_m = (1 - lam) * x + lam * x_b
                x_m = F.dropout(x_m, p=self.dropout, training=self.training)
                
            x_m = self.convs[self.n_layer - 1](x=x_m, edge_index=edge_index, edge_weight=edge_weight, **kwargs)
        
        return x_m


class MmixupGnnModel(GnnModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_classifier(MmixupGNN)
        
    
    def forward(self, x, edge_index, edge_index_b, lam, idx):
        self.output = self.classifier(x, edge_index, edge_index_b, lam, idx)
        return self.output


class Mmixup(mix_base):
    def parse_args(parser):
        parser.add_argument('--tau', type=float, default=1, help='temperature of softmax')
        parser.add_argument('--alpha', type=float, default=4, help='alpha of beta distribution to sample mixup ratio lam')
        
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use(MmixupGnnModel)
    
    
    def epoch_output(self, epoch):
        x, edge_index, y = self.data.x, self.data.edge_index, torch.nn.functional.one_hot(self.data.y, num_classes=self.n_cls).to(torch.float)
        if self.phase == 'train':
            alpha = self.args.alpha
            idx = self.idx()
            idx_train = self.idx('train')
            permuted = torch.randperm(len(idx_train))
            idx[idx_train] = idx[idx_train[permuted]]
        else:
            alpha = 1
            idx = self.idx()
        edge_index_b = self.inv_of_permuted(idx)[edge_index]
        y_b = y[idx]
        
        lam = self.beta(alpha=alpha)
        logits = self.model(x, edge_index, edge_index_b, lam, idx)
        
        return logits, (1 - lam) * y + lam * y_b
    
    
    def logits(self, output):
        logits, _ = output
        y_pred = F.softmax(logits / self.args.tau, dim=-1)
        return y_pred


    def loss(self, output, y):
        logits, y = output
        y_pred = F.softmax(logits / self.args.tau, dim=-1)
        return F.cross_entropy(y_pred, y)
