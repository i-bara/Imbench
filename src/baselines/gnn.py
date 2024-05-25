from .nets import GCNConv, GATConv, SAGEConv
import torch
from torch import nn, optim
from torch.nn import functional as F
from .baseline import Baseline, Timer
import tqdm
import numpy as np
from renode import index2dense


class GNN(nn.Module):
    ''' 
    A GNN backbone, such as GCN, GAT and SAGE.
    '''
    def __init__(self, Conv, n_feat, n_hid, n_cls, dropout, n_layer, encoder=False, **kwargs):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([Conv(n_feat if layer == 0 else n_hid, 
                                         n_cls if layer + 1 == n_layer else n_hid, **kwargs) 
                                         for layer in range(n_layer)])

        self.dropout = dropout
        self.encoder = encoder
        # self.x_dropout = x_dropout
        # self.edge_index_dropout = edge_index_dropout

        self.reg_params = self.convs[:-1].parameters()
        self.non_reg_params = self.convs[-1].parameters()


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
        
        self.reg_params = self.classifier.reg_params
        self.non_reg_params = self.classifier.non_reg_params


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
    

    def _loss(self, output, y, mask, weight, phase, reg):
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
    

    def _output(self, x, edge_index, phase, **kwargs):
        return self.classifier(x=x, edge_index=edge_index, **kwargs)


    def forward(self, x, edge_index, y=None, mask=None, weight=None, logit=False, phase=None, reg=None, **kwargs):
        if mask is None:
            mask = self.baseline.mask()

        output = self._output(x=x, edge_index=edge_index, phase=phase, **kwargs)
        if logit:
            return output
        
        return self._loss(output=output, y=y, mask=mask, weight=weight, phase=phase, reg=reg)


class GnnModelWithEncoder(GnnModel):
    def __init__(self, args, baseline):
        super().__init__(args, baseline)
        self.encoder = GNN(Conv=self.conv_dict[args.net], 
                              n_feat=baseline.n_feat, n_hid=args.feat_dim, n_cls=args.feat_dim, 
                              dropout=args.dropout, 
                              n_layer=1, **self.gnn_kwargs)
        self.classifier = GNN(Conv=self.conv_dict[args.net], 
                              n_feat=args.feat_dim, n_hid=args.feat_dim, n_cls=baseline.n_cls, 
                              dropout=args.dropout, 
                              n_layer=args.n_layer - 1, **self.gnn_kwargs)
        
        self.reg_params = list(self.classifier.reg_params) + list(self.encoder.reg_params)
        self.non_reg_params = list(self.classifier.non_reg_params) + list(self.encoder.non_reg_params)


    def _output(self, x, edge_index, phase, **kwargs):
        if phase == 'embed':
            return self.encoder(x=x, edge_index=edge_index, **kwargs)
        else:
            return self.classifier(x=x, edge_index=edge_index, **kwargs)


class gnn(Baseline):
    def parse_args(parser):
        pass
        # Baseline.add_argument(parser, "--x_dropout", type=float, default=0, help="")
        # Baseline.add_argument(parser, "--edge_index_dropout", type=float, default=0, help="")


    def use(self, Model, *args):
        self.model = Model(args=self.args, baseline=self).to(self.device)
        self.models = [self.model]
        params_dicts = [dict(params=self.model.reg_params, weight_decay=self.args.weight_decay), 
                        dict(params=self.model.non_reg_params, weight_decay=0),]
        for arg in args:
            params_dicts.append(dict(params=arg.parameters(), weight_decay=self.args.weight_decay))
            self.models.append(arg)
        self.optimizer = optim.Adam(params_dicts, lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100)


    def __init__(self, args):
        super().__init__(args)
        self.use(GnnModel)
        self.forward_kwargs = dict()
        self.before_hooks = list()
        self.after_hooks = list()
        self.epoch_timer = Timer(self)
        self.epoch_timer2 = Timer(self)
        self.t = Timer(self)
        self.epoch_time_stat = 0


    def train(self):
        for epoch in tqdm.tqdm(range(self.args.epoch)):
            if self.epoch_time_stat >= 2:
                self.epoch_timer.begin()
            if self.epoch_time_stat >= 1:
                self.epoch_timer2.begin()
            for hook in self.before_hooks:
                hook(epoch=epoch)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.end(f'before: {epoch}')
            self.train_epoch(epoch=epoch)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.tick(f'train : {epoch}')
            self.val_epoch(epoch=epoch)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.tick(f'val   : {epoch}')
            output = self.epoch_loss(epoch=epoch, mode='test')
            self.test(output)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.tick(f'test  : {epoch}')
            for hook in self.after_hooks:
                hook(epoch=epoch)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.end(f'after : {epoch}')
            if self.epoch_time_stat >= 1:
                self.epoch_timer2.end(f'total : {epoch}')

        output = self.epoch_loss(epoch=epoch, mode='test')
        self.test(output)
        return output


    def train_epoch(self, epoch):
        for model in self.models:
            model.train()
        self.optimizer.zero_grad()
        if self.epoch_time_stat >= 3:
            self.t.begin()
        loss = self.train_epoch_loss(epoch=epoch)
        if self.epoch_time_stat >= 3:
            self.t.tick('loss')
        loss.backward()
        if self.epoch_time_stat >= 3:
            self.t.tick('backward')
        self.optimizer.step()
        if self.epoch_time_stat >= 3:
            self.t.tick('step')


    @torch.no_grad()
    def val_epoch(self, epoch):
        for model in self.models:
            model.eval()
        loss = self.val_epoch_loss(epoch=epoch)
        self.scheduler.step(loss)


    def train_epoch_loss(self, epoch):
        return self.epoch_loss(epoch=epoch, mode='train')
    

    def val_epoch_loss(self, epoch):
        return self.epoch_loss(epoch=epoch, mode='val')


    def epoch_loss(self, epoch, mode='test'):
        if mode == 'test':
            return self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, logit=True, **self.forward_kwargs)
        else:
            return self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask(mode), **self.forward_kwargs)
        

    def pr(self, edge_index, pagerank_prob=0.85, limit=None, k=None, eps=None):
        ## ReNode method ##
        ## hyperparam ##

        # calculating the Personalized PageRank Matrix
        pr_prob = 1 - pagerank_prob
        try:
            A = index2dense(edge_index, self.n_sample)
        except IndexError:
            A = edge_index

        if limit is not None and self.n_sample >= limit:  # pagerank limit proposed by PASTEL
            A_hat = A.to(self.device) + torch.eye(A.size(0)).to(self.device)
            D = torch.sum(A_hat, 1)
            D_inv = torch.eye(self.n_sample).to(self.device)

            for iter in range(self.n_sample):
                if (D[iter] == 0):
                    D[iter] = 1e-12
                D_inv[iter][iter] = 1.0 / D[iter]
            D = D_inv.sqrt().to(self.device)

            A_hat = torch.mm(torch.mm(D, A_hat), D)
            temp_matrix = torch.eye(A.size(0)).to(self.device) - pagerank_prob * A_hat
            temp_matrix = temp_matrix.cpu().numpy()
            temp_matrix_inv = np.linalg.inv(temp_matrix).astype(np.float32)

            inv = torch.from_numpy(temp_matrix_inv).to(self.device)
            Pi = pr_prob * inv
        else:  # pagerank implemented by renode
            A_hat   = A.to(self.device) + torch.eye(A.size(0)).to(self.device) # add self-loop
            D       = torch.diag(torch.sum(A_hat,1))
            D       = D.inverse().sqrt()
            A_hat   = torch.mm(torch.mm(D, A_hat), D)
            Pi = pr_prob * ((torch.eye(A.size(0)).to(self.device) - (1 - pr_prob) * A_hat).inverse())
        # Pi = Pi.cpu()

        # two function from SHA to simplify self.Pi, e.g. self.Pi = get_top_k_matrix(self.Pi)
        def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
            num_nodes = A.shape[0]
            row_idx = np.arange(num_nodes)
            A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
            norm = A.sum(axis=0)
            norm[norm <= 0] = 1 # avoid dividing by zero
            return A/norm

        def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
            num_nodes = A.shape[0]
            A[A < eps] = 0.
            norm = A.sum(axis=0)
            norm[norm <= 0] = 1 # avoid dividing by zero
            return A/norm
        
        if k is not None: 
            Pi = get_top_k_matrix(Pi, k=k)

        if eps is not None:
            Pi = get_clipped_matrix(Pi, eps=eps)

        return Pi


    def assert_almost_equal(self, a, b, eps=1e-3):
        assert torch.all((a - b < eps) & (a - b < eps))
