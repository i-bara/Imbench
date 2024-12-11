from .mix_base import mix_base
from .gnnv3 import GNN

import torch
from torch.nn import functional as F


class GraphMixGNN(GNN):
    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # x = F.dropout(x, p=self.x_dropout, training=self.training)
        # edge_index = F.dropout(edge_index, p=self.edge_index_dropout, training=self.training)

        if self.encoder:
            for conv in self.convs:
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)  # is_add_self_loops=self.is_add_self_loops,
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for conv in self.convs[:-1]:
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)
            # , edge_index
        return x


class GraphMix(mix_base):
    def parse_args(parser):
        parser.add_argument('--tau', type=float, default=2, help='temperature of softmax')
        parser.add_argument('--alpha', type=float, default=100, help='alpha of beta distribution to sample mixup ratio lam')
    
    
    def epoch_output(self, epoch):
        if self.epoch % 2 == 0:
            return super().epoch_output(epoch)
        original_y = torch.nn.functional.one_hot(self.data.y, num_classes=self.n_cls).to(torch.float)
        y = torch.zeros_like(original_y, dtype=torch.float, device=self.device)
        y[self.mask('train')] = original_y[self.mask('train')]
        
        k = 10
        y_preds = torch.zeros((k, y.shape[0], y.shape[1]), dtype=y.dtype, device=self.device)
        for i in range(k):
            y_preds[i, :, :] = F.softmax(self.model(x=self.data.x, edge_index=self.data.edge_index) / \
                self.args.tau, dim=-1).detach()
        
        y_pred = y_preds.mean(dim=0)[self.mask('val', 'test')]
        temperature = 0.1
        prob = torch.pow(y_pred, 1.0 / temperature)
        
        y[self.mask('val', 'test')] = prob / prob.sum(dim=1, keepdim=True)
        
        idx_train = self.idx('train')
        idx_val_and_test = self.idx('val', 'test')
        idx_unlabeled = idx_val_and_test[torch.randint(0, len(idx_val_and_test), size=(len(idx_train),))]
        permuted = torch.randperm(len(idx_train))
        
        lam = self.beta()
        x = self.data.x.clone()
        x[idx_train] = (1 - lam) * x[idx_train] + lam * x[idx_train[permuted]]
        y[idx_train] = (1 - lam) * y[idx_train] + lam * y[idx_train[permuted]]
        
        x[idx_unlabeled] = (1 - lam) * x[idx_unlabeled] + lam * x[idx_unlabeled[permuted]]
        y[idx_unlabeled] = (1 - lam) * y[idx_unlabeled] + lam * y[idx_unlabeled[permuted]]
        
        logits = self.model(x, edge_index=self.data.edge_index)
        
        return logits, y, idx_train, idx_unlabeled
    
    
    def logits(self, output):
        if self.epoch % 2 == 0:
            return super().logits(output)
        logits, _, _, _ = output
        y_pred = F.softmax(logits / self.args.tau, dim=-1)
        return y_pred


    def loss(self, output, y):
        if self.epoch % 2 == 0:
            return super().loss(output, y)
        logits, y, idx_train, idx_unlabeled = output
        return F.binary_cross_entropy(F.softmax(logits[idx_train]), y[idx_train]) + \
            F.binary_cross_entropy(F.softmax(logits[idx_unlabeled]), y[idx_unlabeled])
