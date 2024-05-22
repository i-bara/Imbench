from nets import create_gcn, create_sage, create_gat
import torch
from torch import nn, optim
from torch.nn import functional as F
from .baseline import Baseline
import tqdm


class GnnModel(nn.Module):
    def __init__(self, args, baseline):
        super(GnnModel, self).__init__()
        self.args = args
        self.baseline = baseline
        self.criterion_dict = dict()
        if args.net == 'GCN':
            self.classifier = create_gcn(nfeat=baseline.n_feat, nhid=args.feat_dim, nclass=baseline.n_cls, dropout=args.dropout, nlayer=args.n_layer)
        elif args.net == 'GAT':
            self.classifier = create_gat(nfeat=baseline.n_feat, nhid=args.feat_dim, nclass=baseline.n_cls, dropout=args.dropout, nlayer=args.n_layer)
        elif args.net == "SAGE":
            self.classifier = create_sage(nfeat=baseline.n_feat, nhid=args.feat_dim, nclass=baseline.n_cls, dropout=args.dropout, nlayer=args.n_layer)


    def criterion(self, output, y, mask, weight):
        print('loss')
        return F.cross_entropy(output[mask], y[mask], weight=weight)


    def forward(self, x, edge_index, y=None, mask=None, weight=None, logit=False, phase=None):
        if mask is None:
            mask = self.baseline.mask()
        
        output = self.classifier(x=x, adj=edge_index)
        if logit:
            return output
        if phase is None:
            criterion = self.criterion
        else:
            criterion = self.criterion_dict[phase]
        loss = criterion(output=output, y=y, mask=mask, weight=weight)
        return loss


class gnn(Baseline):
    def use(self, Model, *args):
        self.model = Model(args=self.args, baseline=self).to(self.device)
        self.models = [self.model]
        params_dicts = [dict(params=self.model.classifier.reg_params, weight_decay=self.args.weight_decay), 
                        dict(params=self.model.classifier.non_reg_params, weight_decay=0),]
        for arg in args:
            params_dicts.append(dict(params=arg.parameters(), weight_decay=self.args.weight_decay))
            self.models.append(arg)
        self.optimizer = optim.Adam(params_dicts, lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100)


    def __init__(self, args):
        super().__init__(args)
        self.use(GnnModel)


    def train(self):
        for epoch in tqdm.tqdm(range(self.args.epoch)):
            self.train_epoch(epoch=epoch)
            self.val_epoch(epoch=epoch)
            output = self.model(x=self.data.x, edge_index=self.data.edge_index, logit=True)
            self.test(output)

        output = self.model(x=self.data.x, edge_index=self.data.edge_index, logit=True)
        self.test(output)
        return output


    def train_epoch(self, epoch):
        for model in self.models:
            model.train()
        self.optimizer.zero_grad()
        loss = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask('train'))
        loss.backward()
        self.optimizer.step()
        

    @torch.no_grad()
    def val_epoch(self, epoch):
        for model in self.models:
            model.eval()
        loss = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask('val'))
        self.scheduler.step(loss)
