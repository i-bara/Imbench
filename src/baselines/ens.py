from .nets.sha import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModel, GNN
import torch
from torch.nn import functional as F
from gens import neighbor_sampling_ens, duplicate_neighbor, saliency_mixup_ens, sampling_idx_individual_dst, MeanAggregation_ens


class EnsGNN(GNN):
    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # x = F.dropout(x, p=self.x_dropout, training=self.training)
        # edge_index = F.dropout(edge_index, p=self.edge_index_dropout, training=self.training)

        if self.encoder:
            for conv in self.convs:
                x, edge_index = conv(x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for conv in self.convs[:-1]:
                x, edge_index = conv(x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            x, edge_index = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight, **kwargs)
            # , edge_index
        return x


class EnsModel(GnnModel):
    def __init__(self, args, baseline):
        super().__init__(args, baseline)
        self.config_classifier(EnsGNN)


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


class ens(gnn):
    def parse_args(parser):
        parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
        parser.add_argument('--keep_prob', type=float, default=0.01, help='keeping probability') # used in ens
        parser.add_argument('--tau', type=int, default=2, help='temperature in the softmax function when calculating confidence-based node hardness')


    def __init__(self, args):
        super().__init__(args)
        self.use(EnsModel)
        # exit()
        # from neighbor_dist import get_PPR_adj
        # self.Pi2 = get_PPR_adj(self.data.x, self.data.edge_index, alpha=0.05, k=128, eps=None)
        # self.assert_almost_equal(self.Pi, self.Pi2, 0.1)
        # self.Pi = get_PPR_adj(self.data.x, self.data.edge_index, alpha=0.05, k=128, eps=None)

        self.saliency = None
        self.prev_out = None

        self.aggregator = MeanAggregation_ens()


    def init(self):
        self.class_num_list = [self.num(mask=self.mask('train') & self.mask(c)) for c in range(self.n_cls)]  # same as follow
        # self.class_num_list = self.num_list('train').cpu()
        self.idx_info = [self.idx(mask=self.mask('train') & self.mask(c)) for c in range(self.n_cls)]

        self.train_node_mask = self.mask(['train', 'val', 'test'])

        self.Pi = self.pr(self.data.edge_index, pagerank_prob=0.95, k=128, 
                        #   device='cpu' if self.args.dataset == 'ogbn-arxiv' else None, 
                          iterations=40 if self.args.dataset == 'ogbn-arxiv' else None,
                          sparse=self.args.dataset == 'ogbn-arxiv',
                          )


    def backward_hook(self, module, grad_input, grad_output):
        self.saliency = grad_input[0].data


    def mixup(self, x, edge_index, y, epoch):
        self.model.classifier.convs[0].temp_weight.register_backward_hook(self.backward_hook)

        # Sampling source and destination nodes
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(self.class_num_list, self.idx_info, self.device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        ori_saliency = self.saliency[:self.n_sample] if (self.saliency != None) else None

        # Augment nodes
        if epoch > self.args.warmup:
            with torch.no_grad():
                self.prev_out = self.aggregator(self.prev_out, edge_index)  # RuntimeError: CUDA error: device-side assert triggered
                self.prev_out = F.softmax(self.prev_out / self.args.tau, dim=1).detach().clone()
            new_edge_index, dist_kl = neighbor_sampling_ens(self.n_sample, edge_index, sampling_src_idx, sampling_dst_idx, 
                                                            self.Pi, self.prev_out, self.train_node_mask,
                                                            iterations=True)
            new_x = saliency_mixup_ens(x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=self.args.keep_prob)
        else:
            new_edge_index = duplicate_neighbor(self.n_sample, edge_index, sampling_src_idx)
            dist_kl, ori_saliency = None, None
            new_x = saliency_mixup_ens(x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl)
        new_x.requires_grad = True

        new_y = y[sampling_src_idx].clone()
        new_y = torch.cat((y, new_y), dim=0)

        x, edge_index, y = new_x, new_edge_index, new_y

        return x, edge_index, y
    

    def adjust_output(self, output, epoch):
        return output


    def epoch_loss(self, epoch, mode='test'):
        if mode == 'train':
            # embed, edge_index, y = embed, self.data.edge_index, self.data_original.y
            x, edge_index, y = self.mixup(self.data.x, self.data.edge_index, self.data.y, epoch=epoch)

            self.data.x = x
            self.data.edge_index = edge_index
            self.data.y = y
            n_sample = x.shape[0]
            for name, mask in self.masks.items():
                if name == 'train':
                    self.masks[name] = torch.cat((mask, torch.ones(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)
                else:
                    self.masks[name] = torch.cat((mask, torch.zeros(n_sample - self.n_sample, dtype=torch.bool, device=self.device)), dim=0)
        else:
            x, edge_index, y = self.data.x, self.data.edge_index, self.data.y
        if mode == 'test':
            return self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, logit=True, **self.forward_kwargs)
        else:
            if mode == 'train':
                output = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, logit=True, **self.forward_kwargs)
                output = self.adjust_output(output=output, epoch=epoch)
                self.prev_out = (output[:self.n_sample]).clone().detach()
            return self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask(mode), **self.forward_kwargs)
