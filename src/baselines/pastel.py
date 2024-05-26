import math
from .baseline import Baseline
from .gnn import gnn, GnnModel
from .nets.pastel import GCNConv
from .nets import GATConv, SAGEConv  #, GCNConv
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import networkx as nx
import multiprocessing as mp
from torch_geometric.utils import add_self_loops, remove_self_loops
import scipy.sparse as sp


class PastelModel(GnnModel):
    def __init__(self, args, baseline):
        super().__init__(args, baseline)
        self.criterion_default = [self.criterion, self.graph_regularization]
        self.criterion_dict = {
            'graph_learn': {
                self.criterion: 1,
                self.graph_regularization: 1,
                self.graph_learn_regularization: self.args.graph_learn_ratio,
            }
        }
        self.regularizations.add(self.graph_learn_regularization)
        self.regularizations.add(self.graph_regularization)


    def config_gnn(self):
        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
        }
        gnn_kwargs = dict()
        return conv_dict, gnn_kwargs


    def graph_regularization(self, x, cur_raw_adj, **kwargs):
        cur_raw_adj = cur_raw_adj - torch.min(cur_raw_adj)  # The first time may be minus
        graph_loss = 0
        L = torch.diagflat(torch.sum(cur_raw_adj, -1)) - cur_raw_adj
        graph_loss += self.args.smoothness_ratio * torch.trace(torch.mm(x.transpose(-1, -2), torch.mm(L, x))) / int(np.prod(cur_raw_adj.shape))
        # ones_vec = torch.ones(cur_raw_adj.size(-1), device=self.args.device)
        # graph_loss += -self.args.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(cur_raw_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / cur_raw_adj.shape[-1]
        graph_loss -= self.args.degree_ratio * cur_raw_adj.sum(dim=1).add_(1e-12).log().sum(dim=0)
        graph_loss += self.args.sparsity_ratio * torch.sum(torch.pow(cur_raw_adj, 2)) / int(np.prod(cur_raw_adj.shape))
        return graph_loss


    def graph_learn_regularization(self, cur_raw_adj, pre_raw_adj, **kwargs):
        X = cur_raw_adj - pre_raw_adj
        return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))


class pastel(gnn):
    def parse_args(parser):
        Baseline.add_argument(parser, "--x_dropout", type=float, default=0.5, help="")
        Baseline.add_argument(parser, "--raw_adj_dropout", type=float, default=0.5, help="")
        Baseline.add_argument(parser, "--adj_dropout", type=float, default=0.5, help="")

        Baseline.add_argument(parser, "--no_graph_learn", action='store_true', help='whether to debug')
        Baseline.add_argument(parser, "--graph_learn_ratio", type=float, default=0, help="")

        Baseline.add_argument(parser, '--warmup', type=int, default=400, help='warmup epoch')  # 250
        Baseline.add_argument(parser, "--pe_every_epochs", type=int, default=100, help="")  # 50
        Baseline.add_argument(parser, "--gpr_every_epochs", type=int, default=100, help="")  # 50
        Baseline.add_argument(parser, "--iters", type=float, default=1, help="")  # 10

        Baseline.add_argument(parser, "--graph_learn_hidden_size", type=float, default=70, help="")
        Baseline.add_argument(parser, "--graph_learn_topk", type=float, default=None, help="")
        Baseline.add_argument(parser, "--graph_learn_epsilon", type=float, default=0, help="")
        Baseline.add_argument(parser, "--graph_learn_hidden_size2", type=float, default=70, help="")
        Baseline.add_argument(parser, "--graph_learn_topk2", type=float, default=None, help="")
        Baseline.add_argument(parser, "--graph_learn_epsilon2", type=float, default=0, help="")
        Baseline.add_argument(parser, "--graph_learn_num_pers", type=float, default=1, help="")  # 4

        Baseline.add_argument(parser, "--graph_skip_conn", type=float, default=0.8, help="")
        Baseline.add_argument(parser, "--no_graph_include_self", action='store_true', help='whether to debug')

        Baseline.add_argument(parser, "--smoothness_ratio", type=float, default=0.2, help="")
        Baseline.add_argument(parser, "--degree_ratio", type=float, default=0, help="")
        Baseline.add_argument(parser, "--sparsity_ratio", type=float, default=0, help="")

        Baseline.add_argument(parser, "--eps_adj", type=float, default=4e-5, help="")
        Baseline.add_argument(parser, "--update_adj_ratio", type=float, default=0.1, help="")


    def __init__(self, args):
        super().__init__(args)
        self.before_hooks = [self.set_shortest_path]

        if not self.args.no_graph_learn:
            self.graph_learner = GraphLearner(input_size=self.n_feat,
                                              hidden_size=self.args.graph_learn_hidden_size,
                                              n_nodes=self.n_sample,
                                              n_class=self.n_cls,
                                              n_anchors=self.num('train'),
                                              topk=self.args.graph_learn_topk,
                                              epsilon=self.args.graph_learn_epsilon,
                                              n_pers=self.args.graph_learn_num_pers,
                                              device=self.args.device)

            self.graph_learner2 = GraphLearner(input_size=self.n_feat,
                                               hidden_size=self.args.graph_learn_hidden_size2,
                                               n_nodes=self.n_sample,
                                               n_class=self.n_cls,
                                               n_anchors=self.num('train'),
                                               topk=self.args.graph_learn_topk2,
                                               epsilon=self.args.graph_learn_epsilon2,
                                               n_pers=self.args.graph_learn_num_pers,
                                               device=self.args.device)

        else:
            self.graph_learner = None
            self.graph_learner2 = None

        self.use(PastelModel)

        def index2dense(edge_index, nnode=2708):
            idx = edge_index.cpu().numpy()
            adj = np.zeros((nnode,nnode))
            adj[(idx[0], idx[1])] = 1
            return adj
        
        def normalize_sparse_adj(mx):
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        adj = index2dense(self.data.edge_index, self.data.num_nodes)
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize_sparse_adj(adj)
        adj = torch.Tensor(adj.todense()).to(self.device)
        self.adj = adj
        self.cur_adj = adj
        self.group_pagerank_before = self.g(self.pr(adj))
        self.update_gpr(0)
        self.after_hooks = [self.update_gpr]
        self.epoch_time_stat = 3


    def cal_spd(self):
        # num_anchors = self.num('train')
        # num_nodes = self.n_sample
        # spd_mat = np.zeros((num_nodes, num_anchors))

        shortest_path_distance_mat = torch.from_numpy(self.shortest_path_dists).to(self.device).to(torch.float32)  # Use tensor instead of numpy
        # shortest_path_distance_mat = self.shortest_path_dists

        # for iter1 in range(num_nodes):
        #     for iter2 in range(num_anchors):
        #         spd_mat[iter1][iter2] = shortest_path_distance_mat[iter1][self.anchor_node_list[iter2]]

        spd_mat = shortest_path_distance_mat[:, self.mask('train')]

        max_spd = torch.max(spd_mat).item()
        spd_mat = spd_mat / max_spd

        return spd_mat


    def cal_shortest_path_distance(self, adj, approximate):
        n_nodes = self.n_sample
        Adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(Adj)
        G.edges(data=True)
        dists_array = np.zeros((n_nodes, n_nodes))
        dists_dict = all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)

        cnt_disconnected = 0

        for i, node_i in enumerate(G.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(G.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist == -1:
                    cnt_disconnected += 1
                if dist != -1:
                    dists_array[node_i, node_j] = dist
        return dists_array


    def set_shortest_path(self, epoch):
        if epoch % self.args.pe_every_epochs == 0:
            self.position_flag = 1
            self.shortest_path_dists = self.cal_shortest_path_distance(self.cur_adj, 5)
            self.shortest_path_dists_anchor = self.cal_spd()
        else:
            self.position_flag = 0
        

    def rank_group_pagerank(self, pagerank_before, pagerank_after):
        pagerank_dist = torch.mm(pagerank_before, pagerank_after.transpose(-1, -2)).detach().cpu()
        idx = torch.argsort(pagerank_dist.view(-1), descending=True, stable=True)
        inv_idx = torch.zeros(idx.shape, dtype=idx.dtype, device=idx.device)
        inv_idx[idx] = torch.arange(idx.shape[0])
        node_pair_group_pagerank_mat = inv_idx.view(pagerank_dist.shape)

        # num_nodes = self.n_sample
        # node_pair_group_pagerank_mat = np.zeros((num_nodes, num_nodes))
        # node_pair_group_pagerank_mat_list = []
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         node_pair_group_pagerank_mat_list.append(pagerank_dist[i, j])
        # node_pair_group_pagerank_mat_list = np.array(node_pair_group_pagerank_mat_list)
        # index = np.argsort(-node_pair_group_pagerank_mat_list, kind='stable')  # use stable to debug
        # rank = np.argsort(index, kind='stable')

        # rank = rank + 1
        # iter = 0
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         node_pair_group_pagerank_mat[i][j] = rank[iter]
        #         iter = iter + 1

        # self.assert_almost_equal(a, node_pair_group_pagerank_mat)  # In order to assert, set torch.argsort stable=True first!

        return node_pair_group_pagerank_mat


    def cal_group_pagerank_args(self, pagerank_before, pagerank_after):
        node_pair_group_pagerank_mat = self.rank_group_pagerank(pagerank_before, pagerank_after) # rank

        node_pair_group_pagerank_mat = 2 - (torch.cos((node_pair_group_pagerank_mat.to(torch.float32) / (self.n_sample * self.n_sample)) * 3.1415926) + 1)

        # num_nodes = self.n_sample
        # PI = 3.1415926
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         node_pair_group_pagerank_mat[i][j] = 2 - (math.cos((node_pair_group_pagerank_mat[i][j] / (num_nodes * num_nodes)) * PI) + 1)

        # print(node_pair_group_pagerank_mat)
        # print(node_pair_group_pagerank_mat2)
        # self.assert_almost_equal(node_pair_group_pagerank_mat, node_pair_group_pagerank_mat2)

        # exit()

        return node_pair_group_pagerank_mat


    def update_gpr(self, epoch):
        # Calculate group pagerank
        if epoch % self.args.gpr_every_epochs == 0:
            self.group_pagerank_after = self.g(self.pr(self.cur_adj.detach(), device='cpu')).to(self.device)
            self.group_pagerank_args = self.cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after).to(self.device)


    def epoch_loss(self, epoch, mode=None):
        cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner,
                                                self.data.x,
                                                self.shortest_path_dists_anchor,
                                                self.group_pagerank_args,
                                                self.position_flag,
                                                self.args.graph_skip_conn,
                                                graph_include_self=not self.args.no_graph_include_self,
                                                init_adj=self.adj)

        x = F.dropout(self.data.x, self.args.x_dropout, training=self.model.training)
        cur_raw_adj = F.dropout(cur_raw_adj, self.args.raw_adj_dropout, training=self.model.training)
        cur_adj = F.dropout(cur_adj, self.args.adj_dropout, training=self.model.training)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj
        reg = {
            'x': x,
            'cur_raw_adj': cur_raw_adj,
        }

        # loss1 = self.model(x=x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask(mode), reg=reg, **self.forward_kwargs)
        
        loss1 = self.model(x=x, edge_index=self.adj, y=self.data.y, mask=self.mask(mode), reg=reg, **self.forward_kwargs)

        if epoch < self.args.warmup:
            iters = 0
        else:
            iters = self.args.iters  # 10

        eps_adj = self.args.eps_adj
        loss = 0

        for i in range(iters):
            if i > 0 and diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj:
                break

            cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner2,
                                                    self.data.x,
                                                    self.shortest_path_dists_anchor,
                                                    self.group_pagerank_args,
                                                    self.position_flag,
                                                    self.args.graph_skip_conn,
                                                    graph_include_self=not self.args.no_graph_include_self,
                                                    init_adj=self.adj)

            pre_raw_adj, pre_adj = cur_raw_adj, cur_adj

            update_adj_ratio = math.sin(((epoch / self.args.epoch) * 3.1415926)/2) * self.args.update_adj_ratio
            try:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj
            except:
                cur_adj_np = cur_adj.cpu().detach().numpy()
                first_adj_np = first_adj.cpu().detach().numpy()
                cur_adj_np = update_adj_ratio * cur_adj_np + (1 - update_adj_ratio) * first_adj_np
                cur_adj = torch.from_numpy(cur_adj_np).to(self.device)

            reg = {
                'x': x,
                'cur_raw_adj': cur_raw_adj,
                'pre_raw_adj': pre_raw_adj,
            }
            # loss += self.model(x=x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask(mode), reg=reg, **self.forward_kwargs)

            loss += self.model(x=x, edge_index=self.adj, y=self.data.y, mask=self.mask(mode), reg=reg, phase='graph_learn' ,**self.forward_kwargs)

        if iters > 0:
            loss = loss / iters + loss1
        else:
            loss = loss1

        self.cur_adj = cur_adj

        if mode == 'test':
            # output = self.model(x=x, edge_index=self.data.edge_index, y=self.data.y, logit=True, reg=reg, phase='graph_learn' ,**self.forward_kwargs)

            output = self.model(x=x, edge_index=self.adj, y=self.data.y, logit=True, reg=reg, phase='graph_learn' ,**self.forward_kwargs)
            return output        

        return loss


    def learn_graph(self, graph_learner, node_features, position_encoding, gpr_rank, position_flag, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        if not self.args.no_graph_learn:
            raw_adj = graph_learner(node_features, position_encoding, gpr_rank, position_flag)  # most time wasted here (attention bnn)
            #assert raw_adj.min().item() >= 0
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=1e-12)

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + torch.eye(adj.size(0), device=self.device)
            else:
                try:
                    adj.mul_(1 - graph_skip_conn)
                    adj.add_(init_adj.mul(graph_skip_conn))
                except RuntimeError as e:
                    init_adj_cpu = init_adj.to('cpu')
                    adj_cpu = adj.to('cpu')
                    adj_cpu.mul_(1 - graph_skip_conn)
                    adj_cpu.add_(init_adj_cpu.mul(graph_skip_conn))
                    adj = adj_cpu.to('cuda')
            return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj
            return raw_adj, adj


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, n_nodes, n_class, n_anchors, topk=None, epsilon=None, n_pers=16, device=None):
        super(GraphLearner, self).__init__()
        self.n_nodes = n_nodes
        self.n_class = n_class
        self.n_anchors = n_anchors
        self.topk = topk
        self.epsilon = epsilon
        self.device = device
        self.input_size=input_size

        # self.weight_tensor = torch.Tensor(n_pers, input_size)
        # self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

        # self.weight_tensor_for_pe = torch.Tensor(self.n_anchors, hidden_size)
        # self.weight_tensor_for_pe = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_for_pe))

        self.weight_tensor = torch.empty(n_pers, input_size, dtype=torch.float32, device=device)
        nn.init.xavier_uniform_(self.weight_tensor)
        self.weight_tensor = nn.Parameter(self.weight_tensor)

        self.weight_tensor_for_pe = torch.empty(n_anchors, hidden_size, dtype=torch.float32, device=device)
        nn.init.xavier_uniform_(self.weight_tensor_for_pe)
        self.weight_tensor_for_pe = nn.Parameter(self.weight_tensor_for_pe)


    def forward(self, context, position_encoding, gpr_rank, position_flag, ctx_mask=None):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.bmm(context_norm, context_norm.transpose(-1, -2)).mean(0)

        if position_flag == 1:
            pe_fc = torch.mm(position_encoding, self.weight_tensor_for_pe)
            pe_attention = torch.mm(pe_fc, pe_fc.transpose(-1, -2))
            try:
                attention = (attention * 0.5 + pe_attention * 0.5) * gpr_rank
            except RuntimeError as e:
                attention_cpu = attention.to('cpu')
                pe_attention_cpu = pe_attention.to('cpu')
                gpr_rank = gpr_rank.to('cpu')
                attention_cpu = attention_cpu * 0.5 + pe_attention_cpu * 0.5
                attention_cpu = attention_cpu * gpr_rank
                attention = attention_cpu.to('cuda')
        else:
            attention = attention * gpr_rank

        markoff_value = 0

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            if not self.epsilon == 0:
                attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention


    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val).to(self.device)
        return weighted_adjacency_matrix


    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()

        try:
            weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        except:
            attention_np = attention.cpu().detach().numpy()
            mask_np = mask.cpu().detach().numpy()
            weighted_adjacency_matrix_np = attention_np * mask_np + markoff_value * (1 - mask_np)
            weighted_adjacency_matrix = torch.from_numpy(weighted_adjacency_matrix_np).to(self.device)

        return weighted_adjacency_matrix


def diff(X, Y, Z):
    assert X.shape == Y.shape

    try:
        diff_ = torch.sum(torch.pow(X - Y, 2))
        norm_ = torch.sum(torch.pow(Z, 2))
        diff_ = diff_ / torch.clamp(norm_, min=1e-12)
    except:
        X_np = X.cpu().detach().numpy()
        Y_np = Y.cpu().detach().numpy()
        Z_np = Z.cpu().detach().numpy()
        X_Y_np = X_np - Y_np
        X_Y_np_pow = np.power(X_Y_np, 2)
        Z_np_pow = np.power(Z_np, 2)
        diff_np = np.sum(X_Y_np_pow)
        norm_np = np.sum(Z_np_pow)

        diff_ = diff_np / np.clip(a=norm_np, a_min=1e-12, a_max=1e20)

    return diff_


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)   # unweighted
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result
