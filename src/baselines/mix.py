from .nets.sha import GCNConv, GATConv, SAGEConv
from .gnn import gnn, GnnModel, GNN
import torch
from torch import nn, optim
from torch.nn import functional as F
from gens import MeanAggregation_ens
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch, dropout_adj, to_dense_adj
from torch_sparse import SparseTensor
import numpy as np
import random


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


class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, n_cls, dropout, n_layer, encoder=False, **kwargs):
        super(MLP, self).__init__()
        self.convs = nn.ModuleList([nn.Linear(n_feat if layer == 0 else n_hid, 
                                         n_cls if layer + 1 == n_layer else n_hid, **kwargs) 
                                         for layer in range(n_layer)])

        self.dropout = dropout
        self.encoder = encoder
        # self.x_dropout = x_dropout
        # self.edge_index_dropout = edge_index_dropout

        self.reg_params = self.convs[:-1].parameters()
        self.non_reg_params = self.convs[-1].parameters()


    def forward(self, x, **kwargs):
        # x = F.dropout(x, p=self.x_dropout, training=self.training)
        # edge_index = F.dropout(edge_index, p=self.edge_index_dropout, training=self.training)

        if self.encoder:
            for conv in self.convs:
                x = conv(x, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for conv in self.convs[:-1]:
                x = conv(x, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            x = self.convs[-1](x, **kwargs)
            # , edge_index
        return x


class Discriminator(nn.Module):
    def __init__(self, feat_dim):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(feat_dim, feat_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b):
        return self.sigmoid(self.bilinear(a, b))


class MixModel(nn.Module):
    def __init__(self, args, baseline, model=None):
        super(MixModel, self).__init__()
        self.args = args
        self.baseline = baseline
    
        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
        }

        if model is not None:
            self.model = True
            self.encoder1 = model
        else:
            self.model = False
            self.encoder1 = EnsGNN(Conv=conv_dict[self.args.backbone], 
                                n_feat=self.baseline.n_feat, n_hid=self.args.feat_dim, n_cls=self.args.feat_dim, 
                                dropout=self.args.dropout, 
                                n_layer=self.args.n_layer,
                                encoder=True)
        
        self.encoder2 = EnsGNN(Conv=conv_dict[self.args.backbone], 
                              n_feat=self.args.feat_dim, n_hid=self.args.feat_dim, n_cls=self.args.feat_dim, 
                              dropout=self.args.dropout, 
                              n_layer=1)

        self.encoder3 = MLP(n_feat=self.args.feat_dim, n_hid=self.args.feat_dim, n_cls=self.args.feat_dim, 
                            dropout=self.args.dropout, 
                            n_layer=2)
        
        self.classifier = EnsGNN(Conv=conv_dict[self.args.backbone], 
                                 n_feat=self.args.feat_dim, n_hid=self.args.feat_dim, n_cls=self.baseline.n_cls, 
                                 dropout=self.args.dropout, 
                                 n_layer=1)
        
        self.discriminator = Discriminator(feat_dim=self.args.feat_dim)
        
        self.reg_params = list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.encoder3.parameters()) + list(self.discriminator.parameters())
        self.non_reg_params = list(self.classifier.parameters())


    def forward(self, x, edge_index, edge_index_hop, y, mask=None, weight=None, **kwargs):
        if mask is None:
            mask = self.baseline.mask()

        if self.model:
            self.embed1 = x
            for conv in self.encoder1.convs[:-1]:
                self.embed1 = conv(x=self.embed1, edge_index=edge_index, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                self.embed1 = F.relu(self.embed1)
                self.embed1 = F.dropout(self.embed1, p=self.encoder1.dropout, training=self.training)
        else:
            self.embed1 = self.encoder1(x=x, edge_index=edge_index, **kwargs)
        self.embed2 = self.encoder2(x=self.embed1, edge_index=edge_index_hop, **kwargs)
        self.embed3 = self.encoder3(x=self.embed1)
        self.score = self.discriminator(self.embed3, self.embed2)
        if self.model:
            self.embed1_bad = x
            for conv in self.encoder1.convs[:-1]:
                self.embed1_bad = conv(x=self.embed1_bad, edge_index=edge_index, **kwargs)  # is_add_self_loops=self.is_add_self_loops, 
                self.embed1_bad = F.relu(self.embed1_bad)
                self.embed1_bad = F.dropout(self.embed1_bad, p=self.encoder1.dropout, training=self.training)
        else:
            self.embed1_bad = self.encoder1(x=x[torch.randperm(x.shape[0])], edge_index=edge_index, **kwargs)
        self.embed2_bad = self.encoder2(x=self.embed1_bad, edge_index=edge_index_hop, **kwargs)
        self.score_bad = self.discriminator(self.embed3, self.embed2_bad)
        self.output = self.classifier(x=self.embed1, edge_index=edge_index, **kwargs)
        
        return F.cross_entropy(self.output[mask], y[mask], weight=weight) - (torch.log(self.score) + torch.log(1 - self.score_bad))


class mix(gnn):
    def parse_args(parser):
        parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
        parser.add_argument('--keep_prob', type=float, default=0.01, help='keeping probability') # used in ens
        parser.add_argument('--tau', type=int, default=2, help='temperature in the softmax function when calculating confidence-based node hardness')
        parser.add_argument('--max', action="store_true", help='synthesizing to max or mean num of training set. default is mean') 
        parser.add_argument('--no_mask', action="store_true", help='whether to mask the self class in sampling neighbor classes. default is mask')


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
        # ENS
        self.class_num_list = [self.num(mask=self.mask('train') & self.mask(c)) for c in range(self.n_cls)]  # same as follow
        # self.class_num_list = self.num_list('train').cpu()
        self.idx_info = [self.idx(mask=self.mask('train') & self.mask(c)) for c in range(self.n_cls)]

        self.train_node_mask = self.mask(['train', 'val', 'test'])

        self.Pi = self.pr(self.data.edge_index, pagerank_prob=0.95, k=128, 
                        #   device='cpu' if self.args.dataset == 'ogbn-arxiv' else None, 
                          iterations=40 if self.args.dataset == 'ogbn-arxiv' else None,
                          sparse=self.args.dataset == 'ogbn-arxiv',
                          )

        # SHA
        self.train_idx = self.idx('train')
        train_idx_list = self.train_idx.cpu().tolist()
        local2global = {i:train_idx_list[i] for i in range(len(train_idx_list))}
        global2local = dict([val, key] for key, val in local2global.items())
        idx_info_list = [item.cpu().tolist() for item in self.idx_info]
        self.idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]

        self.mi_model = MixModel(args=self.args, baseline=self, model=self.model.classifier).to(self.device)
        params_dicts = [dict(params=self.mi_model.reg_params, weight_decay=self.args.weight_decay), 
                        dict(params=self.mi_model.non_reg_params, weight_decay=0),]
        self.mi_optimizer = optim.Adam(params_dicts, lr=self.args.lr)


    def backward_hook(self, module, grad_input, grad_output):
        self.saliency = grad_input[0].data


    @torch.no_grad()
    def sampling_idx_individual_dst(self, class_num_list, sampling_list, idx_info, device):
        new_class_num_list = torch.Tensor(class_num_list).to(device)

        # Compute # of source nodes
        # for cls_idx, samp_num in zip(idx_info, sampling_list):
            # print('A')
            # print(cls_idx, samp_num)
            # print(torch.randint(len(cls_idx),(int(samp_num.item()),)))
        sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
                            for cls_idx, samp_num in zip(idx_info, sampling_list)]
        sampling_src_idx = torch.cat(sampling_src_idx)

        # Generate corresponding destination nodes
        prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
        prob = prob.repeat_interleave(new_class_num_list.long())
        temp_idx_info = torch.cat(idx_info)
        if sampling_src_idx.shape[0] == 0:
            dst_idx = torch.zeros(0, dtype=torch.int64, device=prob.device)
        else:
            dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)

        temp_idx_info = temp_idx_info.to(dst_idx.device) # 4.2: Fixed bug: do not in the same device

        sampling_dst_idx = temp_idx_info[dst_idx]

        # Sorting src idx with corresponding dst idx
        sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
        sampling_dst_idx = sampling_dst_idx[sorted_idx]

        return sampling_src_idx, sampling_dst_idx
    

    def sample(self, class_num_list, sampling_ratio=1.0):
        max_num, n_cls = max(class_num_list), len(class_num_list)
        if not self.args.max: # mean
            max_num = sum(class_num_list) / n_cls
        sampling_list = (max_num * torch.ones(n_cls) - torch.tensor(class_num_list)) * sampling_ratio
        sampling_list[sampling_list < 0] = 0
        return sampling_list


    @torch.no_grad()
    def sampling_node_source(self, class_num_list, sampling_list, prev_out_local, idx_info_local, train_idx):
        prev_out_local = F.softmax(prev_out_local/self.args.tau, dim=1)
        prev_out_local = prev_out_local.cpu()  # Output (cross entropy) of train (with shape [N, C])

        src_idx_all = []
        dst_idx_all = []
        for cls_idx, num in enumerate(sampling_list):  # For each of classes with train_num < mean(train_num)
            num = int(num.item())
            if num <= 0: 
                continue

            # first sampling
            prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze()  # Prediction of train (with shape [N_c])
            if prob.dim() == 0:  # 4.19: Fixed only one train sample in a class
                src_idx_local = torch.zeros(num, dtype=torch.int64)
            else:
                src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True) 
            src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

            # second sampling
            conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] 
            if not self.args.no_mask:
                conf_src[:,cls_idx] = 0
            neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze(dim=1).tolist()  # squeeze only in dim=1!

            # third sampling
            neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
            dst_idx = []
            for i, item in enumerate(neighbor):
                dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
                dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
            dst_idx = torch.tensor(dst_idx).to(src_idx.device)

            src_idx_all.append(src_idx)
            dst_idx_all.append(dst_idx)
        
        if len(src_idx_all) == 0:
            src_idx_all = torch.zeros(0, dtype=torch.int64, device=train_idx.device)
        else:
            src_idx_all = torch.cat(src_idx_all)
        if len(dst_idx_all) == 0:
            dst_idx_all = torch.zeros(0, dtype=torch.int64, device=train_idx.device)
        else:
            dst_idx_all = torch.cat(dst_idx_all)
        
        return src_idx_all, dst_idx_all


    @torch.no_grad()
    def sampling_node_source_new(self, sampling_list, score=None):
        self.prev_out[self.idx('train')] = F.softmax(self.prev_out[self.idx('train')]/self.args.tau, dim=1)

        src_idx_all = []
        dst_idx_all = []
        for c, num in enumerate(sampling_list):  # For each of classes with train_num < mean(train_num)
            num = int(num.item())
            if num <= 0: 
                continue

            idx = self.idx(mask=self.mask('train') & self.mask(c))

            # first sampling
            prob = 1 - self.prev_out[idx, c]
            src_idx = idx[torch.multinomial(prob + 1e-12, num, replacement=True)]

            # second sampling
            conf_src = self.prev_out[src_idx] 
            if not self.args.no_mask:
                conf_src[:, c] = 0
            neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze(dim=1)  # squeeze only in dim=1!

            # third sampling
            dst_idx = torch.zeros(src_idx.shape, dtype=src_idx.dtype, device=src_idx.device)
            for i in range(num):
                idx_ = self.idx(mask=self.mask('train') & self.mask(neighbor_cls[i].item()))
                dst_idx[i] = torch.multinomial(self.prev_out[idx_, c] + 1e-12, 1)

            src_idx_all.append(src_idx)
            dst_idx_all.append(dst_idx)
        
        if len(src_idx_all) == 0:
            src_idx_all = torch.zeros(0, dtype=torch.int64, device=self.device)
        else:
            src_idx_all = torch.cat(src_idx_all)
        if len(dst_idx_all) == 0:
            dst_idx_all = torch.zeros(0, dtype=torch.int64, device=self.device)
        else:
            dst_idx_all = torch.cat(dst_idx_all)
        
        return src_idx_all, dst_idx_all


    def saliency_mixup_ens(self, x, sampling_src_idx, sampling_dst_idx, lam, saliency=None,
                   dist_kl = None, keep_prob = 0.3):
        """
        Saliency-based node mixing - Mix node features
        Input:
            x:                  Node features; [# of nodes, input feature dimension]
            sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
            sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
            lam:                Sampled mixing ratio; [# of augmented nodes, 1]
            saliency:           Saliency map of input feature; [# of nodes, input feature dimension]
            dist_kl:             KLD between source node and target node predictions; [# of augmented nodes, 1]
            keep_prob:          Ratio of keeping source node feature; scalar
        Output:
            new_x:              [# of original nodes + # of augmented nodes, feature dimension]
        """
        total_node = x.shape[0]
        ## Mixup ##
        new_src = x[sampling_src_idx.to(x.device), :].clone()
        new_dst = x[sampling_dst_idx.to(x.device), :].clone()
        lam = lam.to(x.device)

        # Saliency Mixup
        if saliency != None:
            node_dim = saliency.shape[1]
            saliency_dst = saliency[sampling_dst_idx].abs()
            saliency_dst += 1e-10
            saliency_dst /= torch.sum(saliency_dst, dim=1).unsqueeze(1)

            K = int(node_dim * keep_prob)
            mask_idx = torch.multinomial(saliency_dst, K)
            lam = lam.expand(-1,node_dim).clone()
            if dist_kl != None: # Adaptive
                kl_mask = (torch.sigmoid(dist_kl/3.) * K).squeeze().long()
                idx_matrix = (torch.arange(K).unsqueeze(dim=0).to(kl_mask.device) >= kl_mask.unsqueeze(dim=1))
                zero_repeat_idx = mask_idx[:,0:1].repeat(1,mask_idx.size(1))
                mask_idx[idx_matrix] = zero_repeat_idx[idx_matrix]

            lam[torch.arange(lam.shape[0]).unsqueeze(1), mask_idx] = 1.
        return lam * new_src + (1-lam) * new_dst
    

    @torch.no_grad()
    def neighbor_sampling_ens(total_node, edge_index, sampling_src_idx, sampling_dst_idx,
            neighbor_dist_list, prev_out, train_node_mask=None, iterations=False):
        """
        Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
        Input:
            total_node:         # of nodes; scalar
            edge_index:         Edge index; [2, # of edges]
            sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
            sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
            neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
            prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
            train_node_mask:    Mask for not removed nodes; [# of nodes]
        Output:
            new_edge_index:     original edge index + sampled edge index
            dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
        """
        def get_dist_kl_ens(prev_out, sampling_src_idx, sampling_dst_idx):
            """
            Compute KL divergence
            """
            device = prev_out.device
            dist_kl = F.kl_div(torch.log(prev_out[sampling_dst_idx.to(device)]), prev_out[sampling_src_idx.to(device)], \
                            reduction='none').sum(dim=1,keepdim=True)
            dist_kl[dist_kl<0] = 0
            return dist_kl

        if iterations:
            device = edge_index.device
            n_candidate = 1
            sampling_src_idx = sampling_src_idx.clone().to(device)
            sampling_dst_idx = sampling_dst_idx.clone().to(device) if prev_out is not None else None

            if prev_out is not None:
                dist_kl = get_dist_kl_ens(prev_out, sampling_src_idx, sampling_dst_idx)
                ratio = F.softmax(torch.cat([dist_kl.new_zeros(dist_kl.size(0), 1), -dist_kl], dim=1), dim=1)
                
                # Process src nodes' distribution
                mixed_neighbor_dist = torch.stack([neighbor_dist_list[idx].to_dense() for idx in sampling_src_idx]) * ratio[:, :1]
                
                # Process dst nodes' distribution inside the loop without changing the original sampling_dst_idx tensor
                for i in range(n_candidate):
                    if sampling_dst_idx.dim() == 1:
                        current_dst_idx = sampling_dst_idx.unsqueeze(-1)  # Temporarily adjust dimensions if needed
                    else:
                        current_dst_idx = sampling_dst_idx

                    dst_dist = torch.stack([neighbor_dist_list[idx].to_dense() for idx in current_dst_idx[:, i]])
                    mixed_neighbor_dist += dst_dist * ratio[:, i+1:i+2]
            else:
                # Handle the case where prev_out is not available
                mixed_neighbor_dist = torch.stack([neighbor_dist_list[idx].to_dense() for idx in sampling_src_idx])

            # Normalize mixed_neighbor_dist
            mixed_neighbor_dist.clamp_(min=0).add_(1e-12)
            mixed_neighbor_dist /= mixed_neighbor_dist.sum(dim=1, keepdim=True)

            # Degree distribution processing
            col = edge_index[1]
            degree = scatter_add(torch.ones_like(col, device=device), col, dim_size=total_node)
            if train_node_mask is None:
                train_node_mask = torch.ones_like(degree, dtype=torch.bool, device=device)
            degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask], dim_size=total_node).type(torch.float32)
            degree_dist /= degree_dist.sum()

            # Sample degree and neighbors
            # prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
            # aug_degree = torch.multinomial(prob, 1).squeeze(dim=1)
            prob = degree_dist.unsqueeze(dim=0)
            aug_degree = torch.multinomial(prob, len(sampling_src_idx), replacement=True).squeeze(dim=0)
            max_degree = degree.max().item() + 1
            new_tgt = torch.multinomial(mixed_neighbor_dist, max_degree, replacement=True)
            new_col = new_tgt[(torch.arange(max_degree, device=device) - aug_degree.unsqueeze(dim=1) < 0)]
            new_row = (torch.arange(len(sampling_src_idx), device=device) + total_node).repeat_interleave(aug_degree)
            
            # Combine edges
            inv_edge_index = torch.stack([new_col, new_row], dim=0)

            return inv_edge_index, dist_kl
        ## Exception Handling ##
        device = edge_index.device
        n_candidate = 1
        sampling_src_idx = sampling_src_idx.clone().to(device)

        # Find the nearest nodes and mix target pool
        if prev_out is not None:
            sampling_dst_idx = sampling_dst_idx.clone().to(device)
            dist_kl = get_dist_kl_ens(prev_out, sampling_src_idx, sampling_dst_idx)
            ratio = F.softmax(torch.cat([dist_kl.new_zeros(dist_kl.size(0),1), -dist_kl], dim=1), dim=1)
            mixed_neighbor_dist = ratio[:,:1] * neighbor_dist_list[sampling_src_idx]
            for i in range(n_candidate):
                mixed_neighbor_dist += ratio[:,i+1:i+2] * neighbor_dist_list[sampling_dst_idx.unsqueeze(dim=1)[:,i]]
        else:
            mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

        # Compute degree
        col = edge_index[1]
        degree = scatter_add(torch.ones_like(col), col)
        if len(degree) < total_node:
            degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
        if train_node_mask is None:
            train_node_mask = torch.ones_like(degree,dtype=torch.bool)
        degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

        # Sample degree for augmented nodes
        prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
        aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
        max_degree = degree.max().item() + 1
        aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

        # Sample neighbors
        new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
        tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
        new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
        new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
        new_row = new_row.repeat_interleave(aug_degree)
        inv_edge_index = torch.stack([new_col, new_row], dim=0)

        return inv_edge_index, dist_kl
    

    @torch.no_grad()
    def duplicate_neighbor(self, total_node, edge_index, sampling_src_idx):
        device = edge_index.device

        # Assign node index for augmented nodes
        row, col = edge_index[0], edge_index[1] 
        row, sort_idx = torch.sort(row)
        col = col[sort_idx] 
        degree = scatter_add(torch.ones_like(row), row)
        new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])
        temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

        # Duplicate the edges of source nodes
        node_mask = torch.zeros(total_node, dtype=torch.bool)
        unique_src = torch.unique(sampling_src_idx)
        node_mask[unique_src] = True 

        node_mask = node_mask.to(row.device) # 4.2 Fixed

        row_mask = node_mask[row] 
        edge_mask = col[row_mask] 
        b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
        if len(unique_src) != 0:
            if edge_mask.shape[0] == 0:  # If no edge here, directly return edge_index
                return edge_index
            edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
            if len(temp[temp!=0]) != edge_dense.shape[0]:
                cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
                cut_temp = temp[temp!=0][:-cut_num]
            else:
                cut_temp = temp[temp!=0]
            edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
            new_col = edge_dense[edge_dense!= -1]
            inv_edge_index = torch.stack([new_col, new_row], dim=0)
            return inv_edge_index
        else:
            return torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)


    def mixup_ens(self, x, edge_index, y, epoch, sampling_list=None):
        self.model.classifier.convs[0].temp_weight.register_backward_hook(self.backward_hook)

        # Sampling source and destination nodes
        if sampling_list is None:
            sampling_list = self.sample(self.class_num_list)
        if type(sampling_list) is float:
            sampling_list = self.sample(self.class_num_list, sampling_list)
        sampling_src_idx, sampling_dst_idx = self.sampling_idx_individual_dst(self.class_num_list, sampling_list, self.idx_info, self.device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        ori_saliency = self.saliency[:self.n_sample] if (self.saliency != None) else None

        # Augment nodes
        if epoch > self.args.warmup:
            with torch.no_grad():
                self.prev_out = self.aggregator(self.prev_out, edge_index)  # RuntimeError: CUDA error: device-side assert triggered
                self.prev_out = F.softmax(self.prev_out / self.args.tau, dim=1).detach().clone()
            new_edge_index, dist_kl = self.neighbor_sampling_ens(self.n_sample, edge_index, sampling_src_idx, sampling_dst_idx, 
                                                            self.Pi, self.prev_out, self.train_node_mask,
                                                            iterations=True)
            new_x = self.saliency_mixup_ens(x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=self.args.keep_prob)
        else:
            new_edge_index = self.duplicate_neighbor(self.n_sample, edge_index, sampling_src_idx)
            dist_kl, ori_saliency = None, None
            new_x = self.saliency_mixup_ens(x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl)
        new_x.requires_grad = True

        new_y = y[sampling_src_idx].clone()

        return new_x, new_edge_index, new_y
    

    def saliency_mixup_sha(self, x, sampling_src_idx, sampling_dst_idx, lam):
        new_src = x[sampling_src_idx.to(x.device), :].clone()
        new_dst = x[sampling_dst_idx.to(x.device), :].clone()
        lam = lam.to(x.device)

        return lam * new_src + (1-lam) * new_dst
    

    @torch.no_grad()
    def neighbor_sampling_sha(self, total_node, edge_index, sampling_src_idx,
            neighbor_dist_list, train_node_mask=None, iterations=False):
        """
        Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
        Input:
            total_node:         # of nodes; scalar
            edge_index:         Edge index; [2, # of edges]
            sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
            sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
            neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
            prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
            train_node_mask:    Mask for not removed nodes; [# of nodes]
        Output:
            new_edge_index:     original edge index + sampled edge index
            dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
        """
        if iterations:
            device = edge_index.device
            sampling_src_idx = sampling_src_idx.clone().to(device)
            # Ensure neighbor_dist_list is appropriately dimensioned for multinomial
            mixed_neighbor_dist = neighbor_dist_list.to(device).squeeze(-1)  
            #neighbor_dist_list现在是一个ppr向量！！
            # Compute degree
            col = edge_index[1]
            degree = scatter_add(torch.ones_like(col), col)
            if len(degree) < total_node:
                degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))], dim=0)
            if train_node_mask is None:
                train_node_mask = torch.ones_like(degree, dtype=torch.bool)
            degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

            # Sample degree for augmented nodes
            prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
            aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
            max_degree = degree.max().item() + 1
            aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

            # Sample neighbors
            if mixed_neighbor_dist.dim() == 1:
                # Ensure it's treated as a distribution over nodes for each sampling index
                mixed_neighbor_dist = mixed_neighbor_dist.unsqueeze(0).expand(len(sampling_src_idx), -1)
            new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree, replacement=True)
            tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
            valid_indices = (tgt_index - aug_degree.unsqueeze(dim=1) < 0)
            new_col = new_tgt[valid_indices]
            new_row = torch.arange(len(sampling_src_idx)).to(device).repeat_interleave(aug_degree) + total_node
            inv_edge_index = torch.stack([new_col, new_row], dim=0)
            new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
        else:
            ## Exception Handling ##
            device = edge_index.device
            sampling_src_idx = sampling_src_idx.clone().to(device)
            
            # Find the nearest nodes and mix target pool
            mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

            # Compute degree
            col = edge_index[1]
            degree = scatter_add(torch.ones_like(col), col)
            if len(degree) < total_node:
                degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
            if train_node_mask is None:
                train_node_mask = torch.ones_like(degree,dtype=torch.bool)
            degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

            # Sample degree for augmented nodes
            prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
            aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
            max_degree = degree.max().item() + 1
            aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

            # Sample neighbors
            new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
            tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
            new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
            new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
            new_row = new_row.repeat_interleave(aug_degree)
            inv_edge_index = torch.stack([new_col, new_row], dim=0)

        return inv_edge_index


    def mixup_sha(self, x, edge_index, y, epoch, sampling_list=None):
        if sampling_list is None:
            sampling_list = self.sample(self.class_num_list)
        if type(sampling_list) is float:
            sampling_list = self.sample(self.class_num_list, sampling_list)
        
        if epoch > self.args.warmup:
            
            # identifying source samples
            sampling_src_idx, sampling_dst_idx = self.sampling_node_source_new(sampling_list) 
            # sampling_src_idx, sampling_dst_idx = self.sampling_node_source(self.class_num_list, sampling_list, self.prev_out[self.idx('train')], self.idx_info_local, self.idx('train')) 

            # semimxup
            new_edge_index = self.neighbor_sampling_sha(self.n_sample, edge_index, sampling_src_idx, self.Pi, iterations=self.args.dataset == 'ogbn-arxiv')
            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = self.saliency_mixup_sha(x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = self.sampling_idx_individual_dst(self.class_num_list, sampling_list, self.idx_info, self.device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = self.duplicate_neighbor(self.n_sample, edge_index, sampling_src_idx)
            new_x = self.saliency_mixup_sha(x, sampling_src_idx, sampling_dst_idx, lam)

        new_y = y[sampling_src_idx].clone()

        return new_x, new_edge_index, new_y


    def mixup_sha_new(self, x, edge_index, y, epoch, sampling_list=None):
        if sampling_list is None:
            sampling_list = self.sample(self.class_num_list)
        if type(sampling_list) is float:
            sampling_list = self.sample(self.class_num_list, sampling_list)
        
        if epoch > self.args.warmup:
            
            # identifying source samples
            sampling_src_idx, sampling_dst_idx = self.sampling_node_source(self.class_num_list, sampling_list, self.prev_out[self.idx('train')], self.idx_info_local, self.idx('train')) 
            
            # semimxup
            new_edge_index = self.neighbor_sampling_sha(self.n_sample, edge_index, sampling_src_idx, self.Pi, iterations=self.args.dataset == 'ogbn-arxiv')
            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = self.saliency_mixup_sha(x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = self.sampling_idx_individual_dst(self.class_num_list, sampling_list, self.idx_info, self.device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = self.duplicate_neighbor(self.n_sample, edge_index, sampling_src_idx)
            new_x = self.saliency_mixup_sha(x, sampling_src_idx, sampling_dst_idx, lam)

        new_y = y[sampling_src_idx].clone()

        return new_x, new_edge_index, new_y


    def uniform(self, k=1.0):
        return random.random() * k


    def hop(self, x, edge_index, r: int):
        assert r > 0
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
              sparse_sizes=(x.shape[0], x.shape[0])).to_dense()
        new_adj = adj.clone()
        for _ in range(r - 1):
            new_adj = new_adj * adj + new_adj
        new_adj[new_adj >= 1] = 1
        new_edge_index = new_adj.nonzero().t().contiguous()
        return new_edge_index


    def igraphmix(self, x, edge_index, y, lam, src, dst, n_node, connect=True):
        new_x = (1 - lam) * x[src] + lam * x[dst]

        new_y = (1 - lam) * y[src] + lam * y[dst]

        new_edge_index = torch.zeros((2, 0), dtype=edge_index.dtype, device=self.device)

        for idx, p in [(src, lam), (dst, 1 - lam)]:
            print(x.shape[0], idx.shape[0])
            # adj = torch.zeros((x.shape[0] + idx.shape[0], x.shape[0] + idx.shape[0]), dtype=torch.bool, device=self.device)
            
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                   sparse_sizes=(x.shape[0] + idx.shape[0], x.shape[0] + idx.shape[0])).to_dense()
            
            # for i in range(edge_index.shape[1]):
            #     adj[edge_index[0, i], edge_index[1, i]] = True

            adj[x.shape[0]:, :x.shape[0]] = adj[:x.shape[0], :x.shape[0]][idx, :]
            adj[:, x.shape[0]:] = adj[:, :x.shape[0]][:, idx]
            adj[:x.shape[0], :x.shape[0]] = 0
            if not connect:
                adj[x.shape[0]:, :x.shape[0]] = 0
                adj[:x.shape[0], x.shape[0]:] = 0

            new_edge_index = torch.cat((new_edge_index, dropout_adj(adj.nonzero().t().contiguous(), p=p, training=True)[0] + (n_node - x.shape[0])), dim=1)

        print(new_edge_index.shape)


        return new_x, new_edge_index, new_y


    def igraphmix2(self, x, edge_index, y, lam, idx):
        x = (1 - lam) * x + lam * x[idx]

        y = (1 - lam) * y + lam * y[idx]

        edge_index = torch.cat((
            dropout_adj(edge_index, p=lam, training=True)[0], 
            dropout_adj(self.inv2(idx)[edge_index], p=1 - lam, training=True)[0]), dim=1)
        
        return x, edge_index, y


    class MixSmote:
        def mix_x(x, edge_index, y, new_x):
            lam = random.random()
            return x + (new_x - x) * lam
    

    def mixup_smote(self, x, edge_index, y, epoch, sampling_list=None, Mix=MixSmote):
        if sampling_list is None:
                sampling_list = self.sample(self.class_num_list)
        if type(sampling_list) is float:
            sampling_list = self.sample(self.class_num_list, sampling_list)

        chosen = None

        for c in range(self.n_cls):
            num = int(sampling_list[c])
            new_chosen = self.idx(mask=self.mask('train') & self.mask(c))[:num]

            chosen_embed = x[new_chosen,:]
            if chosen_embed.shape[0] != 0:
                from scipy.spatial.distance import pdist, squareform

                distance = squareform(pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance,distance.max()+100)

                idx_neighbor = distance.argmin(axis=-1)
                    
                embed = Mix.mix_x(chosen_embed, edge_index, y, chosen_embed[idx_neighbor,:])

                if chosen is None:
                    chosen = new_chosen
                    new_features = embed
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
                    new_features = torch.cat((new_features, embed),0)

        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                   sparse_sizes=(x.shape[0], x.shape[0]))
        adj_back = adj.to_dense()
        add_num = chosen.shape[0]
        new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
        new_adj[:adj_back.shape[0], :adj_back.shape[0]] = 0
        new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
        new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
        new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]
        adj = new_adj.to_sparse()
        new_edge_index = adj.indices()

        new_x = new_features.clone().detach()
        new_y = y[chosen].clone().detach()
        
        return new_x, new_edge_index, new_y
    

    def mixup_smote_new(self, x, edge_index, y, epoch, sampling_list=None, connect=True, score=None):
        if sampling_list is None:
                sampling_list = self.sample(self.class_num_list)
        if type(sampling_list) is float:
            sampling_list = self.sample(self.class_num_list, sampling_list)

        chosen = torch.zeros((0, ), dtype=torch.int64, device=self.device)
        new_x, new_edge_index, new_y = self.new((x, edge_index, y))

        for c in range(self.n_cls):
            num = int(sampling_list[c])
            if score is None:
                new_chosen = self.idx(mask=self.mask('train') & self.mask(c))[:num]
            else:
                idx_all = self.idx(mask=self.mask('train') & self.mask(c))
                _, indices = torch.sort(score[idx_all], descending=True)
                # indices = torch.squeeze(indices, dim=1)
                new_chosen = idx_all[indices][:num]

            chosen_embed = x[new_chosen,:]
            if chosen_embed.shape[0] != 0:
                from scipy.spatial.distance import pdist, squareform

                distance = squareform(pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance,distance.max()+100)

                idx_neighbor = distance.argmin(axis=-1)

                if connect:
                    this_x, this_edge_index, this_y = self.igraphmix(x, edge_index, y, self.uniform(0.2), new_chosen, new_chosen[idx_neighbor], 
                                                                     n_node=x.shape[0], connect=connect)
                else:
                    this_x, this_edge_index, this_y = self.igraphmix(x, edge_index, y, self.uniform(0.2), new_chosen, new_chosen[idx_neighbor], 
                                                                     n_node=x.shape[0] + new_x.shape[0], connect=connect)

                chosen = torch.cat((chosen, new_chosen))
                new_x, new_edge_index, new_y = self.cat((new_x, new_edge_index, new_y), (this_x, this_edge_index, this_y))
                if connect:
                    x, edge_index, y = self.cat((x, edge_index, y), (this_x, this_edge_index, this_y))
        
        return new_x, new_edge_index, new_y


    def new(self, old):
        x, edge_index, y = old
        x = torch.zeros((0, x.shape[1]), dtype=x.dtype, device=x.device)
        edge_index = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        if len(y.shape) == 1:  # hard label
            y = torch.zeros((0,), dtype=y.dtype, device=y.device)
        else:  # soft label
            y = torch.zeros((0, y.shape[1]), dtype=y.dtype, device=y.device)
        return x, edge_index, y


    def cat(self, old, *args):
        x, edge_index, y = old
        for new in args:
            new_x, new_edge_index, new_y = new
            x = torch.cat((x, new_x), dim=0)
            edge_index = torch.cat((edge_index, new_edge_index), dim=1)
            y = torch.cat((y, new_y), dim=0)
        return x, edge_index, y


    def adjust_output(self, output, epoch):
        return output


    def epoch_loss(self, epoch, mode='test'):
        if mode == 'train':
            # evaluate mutual information score
            self.mi_optimizer.zero_grad()
            x, edge_index, y = self.data.x, self.data.edge_index, self.data.y
            edge_index_hop = self.hop(x, edge_index, 5)
            mi_loss = self.mi_model(x, edge_index, edge_index_hop, y, mask=self.mask(mode))
            # score = torch.squeeze(self.mi_model.score, dim=1)
            # score = torch.max(self.mi_model.output, dim=1).values
            score = torch.max(self.mi_model.output, dim=1).values + torch.squeeze(self.mi_model.score, dim=1)
            print(score.shape, score.dtype)
            print(mi_loss.shape)
            torch.mean(mi_loss).backward()
            self.mi_optimizer.step()

            # embed, edge_index, y = embed, self.data.edge_index, self.data_original.y
            x, edge_index, y = self.data.x, self.data.edge_index, torch.nn.functional.one_hot(self.data.y, num_classes=self.n_cls).to(torch.float)
            x, edge_index, y = self.cat(
                (x, edge_index, y), 
                # self.mixup_smote(x, edge_index, y, epoch=epoch, sampling_list=1.0),
                # self.mixup_smote_new(x, edge_index, y, epoch=epoch, sampling_list=1.0, connect=True),
                # self.mixup_smote_new(x, edge_index, y, epoch=epoch, sampling_list=0.2, connect=True, score=score),
                self.mixup_sha(x, edge_index, y, epoch=epoch, sampling_list=0.5),
                )
            print(x.shape, edge_index.shape, y.shape, edge_index.max().item())

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
