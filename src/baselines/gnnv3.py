from .nets import GCNConv, GATConv, SAGEConv
import torch
from torch import nn, optim
from torch.nn import functional as F
from .baseline import Baseline, Timer
import tqdm
import numpy as np
from renode import index2dense
from torch_geometric.utils import to_torch_csr_tensor
import random


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
        self.conv_dict, self.gnn_kwargs = self.config_gnn()

        self.config_classifier(GNN)


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
    

    def config_classifier(self, n):
        self.classifier = n(Conv=self.conv_dict[self.args.net], 
                              n_feat=self.baseline.n_feat, n_hid=self.args.feat_dim, n_cls=self.baseline.n_cls, 
                              dropout=self.args.dropout, 
                            #   x_dropout=args.x_dropout, 
                            #   edge_index_dropout=args.edge_index_dropout, 
                              n_layer=self.args.n_layer, **self.gnn_kwargs)
        
        self.reg_params = self.classifier.reg_params
        self.non_reg_params = self.classifier.non_reg_params


    def forward(self, x, edge_index, y=None):
        # if mask is None:
        #     mask = self.baseline.mask()

        self.output = self.classifier(x=x, edge_index=edge_index)
        return self.output


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
        self.model = Model(args=self.args, baseline=self).to(self.device)  # A model turns x, edge_index, y into the loss
        self.models = [self.model]  # To set model.train() and model.val()
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
        
        self.minibatch_size = None
        self.args.epoch *= 1 if args.dataset == 'ogbn-arxiv' else 1

        self.save()
        
        if self.minibatch_size is not None:
            self.init_batch(minibatch_size=self.minibatch_size)
        else:
            self.init()

        self.training = False


    def train(self):
        for epoch in tqdm.tqdm(range(self.args.epoch)):
            self.epoch = epoch
            if self.epoch_time_stat >= 2:
                self.epoch_timer.begin()
            if self.epoch_time_stat >= 1:
                self.epoch_timer2.begin()
            for hook in self.before_hooks:
                hook(epoch=epoch)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.end(f'before: {epoch}')

            # train
            self.phase = 'train'
            self.train_epoch(epoch=epoch)

            if self.epoch_time_stat >= 2:
                self.epoch_timer.tick(f'train : {epoch}')

            # val
            self.phase = 'val'
            self.val_epoch(epoch=epoch)

            if self.epoch_time_stat >= 2:
                self.epoch_timer.tick(f'val   : {epoch}')
            logits = self._epoch_logits(epoch=epoch)

            # test
            self.phase = 'test'
            self.test(logits=logits)
            print(f'epoch {epoch}: {self.test_acc * 100:.3f}')

            if self.epoch_time_stat >= 2:
                self.epoch_timer.tick(f'test  : {epoch}')
            for hook in self.after_hooks:
                hook(epoch=epoch)
            if self.epoch_time_stat >= 2:
                self.epoch_timer.end(f'after : {epoch}')
            if self.epoch_time_stat >= 1:
                self.epoch_timer2.end(f'total : {epoch}')

        logits = self._epoch_logits(epoch=epoch)
        self.test(logits=logits)
        return logits


    def train_epoch(self, epoch):
        for model in self.models:
            model.train()
        self.training = True
        self.optimizer.zero_grad()
        if self.epoch_time_stat >= 3:
            self.t.begin()
            
        loss = self._epoch_loss(epoch=epoch)
        
        if loss.dim() == 0:
            pass
        elif loss.dim() == 1:
            loss = loss[self.mask('train')].mean()
        elif loss.dim() == 2:
            assert(loss.shape[1] == 1), "loss must be 1-dim"
            loss = loss[self.mask('train')].mean()
        
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
        self.training = False
        
        loss = self._epoch_loss(epoch=epoch)
        if loss.dim() == 0:
            pass
        elif loss.dim() == 1:
            loss = loss[self.mask('val')].mean()
        elif loss.dim() == 2:
            assert(loss.shape[1] == 1), "loss must be 1-dim"
            loss = loss[self.mask('val')].mean()
        
        self.scheduler.step(loss)
        
        
    def _epoch_logits(self, epoch):
        return self.logits(self.epoch_output(epoch=epoch))


    def _epoch_loss(self, epoch):
        self.restore()
        if self.minibatch_size is not None:
            self.batch()
            self.init()
        return self.loss(self.epoch_output(epoch=epoch), y=self.data.y)


    def init(self):
        pass


    def epoch_output(self, epoch):
        return self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, **self.forward_kwargs)
    
    
    def logits(self, output):
        return output


    def loss(self, output, y):
        return F.cross_entropy(output, y, reduction='none')
        

    def pr(self, edge_index, pagerank_prob=0.85, limit=None, k=None, eps=None, device=None, iterations=None, sparse=False):
        if device is None:
            device = self.device

        ## ReNode method ##
        ## hyperparam ##

        # calculating the Personalized PageRank Matrix
        pr_prob = 1 - pagerank_prob

        A = to_torch_csr_tensor(edge_index=edge_index, size=self.n_sample)
        eye = to_torch_csr_tensor(edge_index=torch.stack((torch.arange(self.n_sample, dtype=torch.int64, device=device), 
                                                          torch.arange(self.n_sample, dtype=torch.int64, device=device)), 
                                                          dim=0), size=self.n_sample)
        # A = csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(self.n_sample, self.n_sample))

        # try:
        #     A = index2dense(edge_index, self.n_sample)
        #     exit()
        # except IndexError:
        #     A = edge_index

        if limit is not None and self.n_sample >= limit:  # pagerank limit proposed by PASTEL
            A_hat = A.to(device) + torch.eye(A.size(0), layout=torch.sparse_csr).to(device)
            D = torch.sum(A_hat, 1)
            D_inv = torch.eye(self.n_sample).to(device)

            for iter in range(self.n_sample):
                if (D[iter] == 0):
                    D[iter] = 1e-12
                D_inv[iter][iter] = 1.0 / D[iter]
            D = D_inv.sqrt().to(device)

            A_hat = torch.mm(torch.mm(D, A_hat), D)
            temp_matrix = torch.eye(A.size(0)).to(device) - pagerank_prob * A_hat
            temp_matrix = temp_matrix.cpu().numpy()
            temp_matrix_inv = np.linalg.inv(temp_matrix).astype(np.float32)

            inv = torch.from_numpy(temp_matrix_inv).to(device)
            Pi = pr_prob * inv
        else:  # pagerank implemented by renode
            A = A.to(device)
            if sparse:
                A_hat = A + eye
                D_ = torch.sqrt(torch.sum(A_hat, dim=1, keepdim=True))
                D = to_torch_csr_tensor(edge_index=torch.stack((torch.arange(self.n_sample, dtype=torch.int64, device=device),   
                                                            torch.arange(self.n_sample, dtype=torch.int64, device=device)), 
                                                            dim=0), 
                                                            edge_attr=1 / D_.values(),
                                                            size=self.n_sample)
                A_hat = D @ A_hat @ D
            else:
                A_hat   = A.to_dense() + torch.eye(A.size(0), device=device) # add self-loop
                D       = torch.diag(torch.sum(A_hat,1))
                D       = D.inverse().sqrt()
                # D       = torch.diag(1 / torch.sqrt(torch.sum(A_hat, dim=1)))
                A_hat   = torch.mm(torch.mm(D, A_hat), D)
            
            # self.assert_almost_equal(A_hat, A_hat_1)
            # try:
            #     if self.nnn == 1:
            #         exit()
            #     else:
            #         self.nnn += 1
            # except AttributeError:
            #     self.nnn = 0
            
            if iterations is not None:
                Pi = torch.ones((A.size(0), 1), dtype=torch.float, device=device) / A.size(0)
                for _ in range(iterations):
                    Pi = pr_prob * A_hat @ Pi + (1 - pr_prob) * torch.ones((A.size(0), 1), dtype=torch.float, device=device) / A.size(0)
            else:
                if sparse:
                    raise NotImplementedError
                else:
                    Pi = pr_prob * ((torch.eye(A.size(0), device=device) - (1 - pr_prob) * A_hat).inverse())  # pastel will be stuck on the inverse() over GPU?
            
            
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

        def get_top_k_iter(A, k):
            sorted_indices = torch.argsort(-A, axis=0)
            top_k = torch.zeros_like(A)
            for col in range(A.shape[1]):
                top_k[sorted_indices[:k, col], col] = A[sorted_indices[:k, col], col]
            norm = top_k.sum(axis=0)
            norm[norm == 0] = 1
            return top_k / norm

        def get_top_k_csr(A, k):
            num_nodes = A.shape[0]
            pass
        
        if k is not None: 
            if iterations is not None:
                Pi = get_top_k_iter(Pi, k=k)
            else:
                if sparse and iterations is None:
                    Pi = get_top_k_csr(Pi, k=k)
                else:
                    Pi = get_top_k_matrix(Pi, k=k)

        if eps is not None:
            Pi = get_clipped_matrix(Pi, eps=eps)

        return Pi


    def g(self, Pi, iterations=False):
        # calculating the ReNode Weight
        gpr_matrix = [] # the class-level influence distribution

        for c in range(self.n_cls):
            #iter_Pi = data.Pi[torch.tensor(target_data.train_node[iter_c]).long()]
            iter_Pi = Pi[(self.mask(c) & self.mask('train')).to(Pi.device)] # check! is it same with above line?
            iter_gpr = torch.mean(iter_Pi,dim=0).squeeze()
            gpr_matrix.append(iter_gpr)

        temp_gpr = torch.stack(gpr_matrix,dim=0)
        if iterations:
            temp_gpr = temp_gpr.unsqueeze(dim=1)
        return temp_gpr.transpose(0,1)


    def assert_almost_equal(self, a, b, eps=1e-3):
        assert torch.all((a - b < eps) & (a - b < eps))
