from .ens import ens
import torch
from gens import sampling_node_source, neighbor_sampling, duplicate_neighbor, saliency_mixup, sampling_idx_individual_dst


class sha(ens):
    def parse_args(parser):
        parser.add_argument('--max', action="store_true", help='synthesizing to max or mean num of training set. default is mean') 
        parser.add_argument('--no_mask', action="store_true", help='whether to mask the self class in sampling neighbor classes. default is mask')
        

    def init(self):
        super().init()
        self.train_idx = self.idx('train')
        train_idx_list = self.train_idx.cpu().tolist()
        local2global = {i:train_idx_list[i] for i in range(len(train_idx_list))}
        global2local = dict([val, key] for key, val in local2global.items())
        idx_info_list = [item.cpu().tolist() for item in self.idx_info]
        self.idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]


    def mixup(self, x, edge_index, y, epoch):
        if epoch > self.args.warmup:
            
            # identifying source samples
            sampling_src_idx, sampling_dst_idx = sampling_node_source(self.class_num_list, self.prev_out[self.train_idx], self.idx_info_local, self.train_idx, self.args.tau, self.args.max, self.args.no_mask) 
            
            # semimxup
            new_edge_index = neighbor_sampling(self.n_sample, edge_index, sampling_src_idx, self.Pi, iterations=self.args.dataset == 'ogbn-arxiv')
            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(self.class_num_list, self.idx_info, self.device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = duplicate_neighbor(self.n_sample, edge_index, sampling_src_idx)
            new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam)

        new_y = y[sampling_src_idx].clone()
        new_y = torch.cat((y, new_y), dim=0)

        x, edge_index, y = new_x, new_edge_index, new_y

        return x, edge_index, y
