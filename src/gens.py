import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, add_self_loops
# smote
from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_sparse import SparseTensor
from copy import deepcopy

@torch.no_grad()
def sampling_idx_individual_dst(class_num_list, idx_info, device):
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    new_class_num_list = torch.Tensor(class_num_list).to(device)

    # Compute # of source nodes
    sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
                        for cls_idx, samp_num in zip(idx_info, sampling_list)]
    sampling_src_idx = torch.cat(sampling_src_idx)

    # Generate corresponding destination nodes
    prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)

    temp_idx_info = temp_idx_info.to(dst_idx.device) # 4.2: Fixed bug: do not in the same device

    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx

def saliency_mixup_ens(x, sampling_src_idx, sampling_dst_idx, lam, saliency=None,
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
    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x

def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x

@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
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
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
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
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

@torch.no_grad()
def neighbor_sampling_ens(total_node, edge_index, sampling_src_idx, sampling_dst_idx,
        neighbor_dist_list, prev_out, train_node_mask=None):
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
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index, dist_kl

@torch.no_grad()
def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False, no_mask=False):
    max_num, n_cls = max(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)

    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
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
        if not no_mask:
            conf_src[:,cls_idx] = 0
        neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist() 

        # third sampling
        neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
        dst_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
            dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all

class MeanAggregation_ens(MessagePassing):
    def __init__(self):
        super(MeanAggregation_ens, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

def get_dist_kl_ens(prev_out, sampling_src_idx, sampling_dst_idx):
    """
    Compute KL divergence
    """
    device = prev_out.device
    dist_kl = F.kl_div(torch.log(prev_out[sampling_dst_idx.to(device)]), prev_out[sampling_src_idx.to(device)], \
                    reduction='none').sum(dim=1,keepdim=True)
    dist_kl[dist_kl<0] = 0
    return dist_kl

def src_smote(data, portion=1.0, im_class_num=3):
    features = data.x
    labels = data.y

    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                   sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    # adj = to_scipy_sparse_matrix(data.edge_index)  Doesn't work because of scipy instead of tensor

    idx_train = torch.tensor(range(data.train_mask.shape[0]), device=data.train_mask.device)[data.train_mask]

    # print(data.train_mask.shape)
    # print(data.train_mask[range(data.train_mask.shape[0])].shape)

    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        # print(idx_train)
        # print((labels==(c_largest-i)))
        # print((labels==(c_largest-i))[idx_train])
        # print(idx_train.shape)
        # print((labels==(c_largest-i)).shape)
        # print((labels==(c_largest-i))[idx_train].shape)
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        # print('chosen: %d' % new_chosen.shape)
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            # print(new_chosen, new_chosen.shape)
            new_chosen = new_chosen[:num]
            # print(new_chosen, new_chosen.shape)

            chosen_embed = features[new_chosen,:]
            if chosen_embed.shape[0] != 0:
                distance = squareform(pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance,distance.max()+100)

                idx_neighbor = distance.argmin(axis=-1)
                
                interp_place = random.random()
                embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

                if chosen is None:
                    chosen = new_chosen
                    new_features = embed
                else:
                    # print(new_features.shape)
                    # print(chosen.shape)
                    chosen = torch.cat((chosen, new_chosen), 0)
                    new_features = torch.cat((new_features, embed),0)
            
        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        if chosen_embed.shape[0] != 0:
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
                
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                # print(new_features.shape)
                # print(chosen.shape)
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    # print(new_features.shape)
    # print(labels.shape)
    # print(chosen.shape)

    #ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    # print('aaaaaaaa')
    # print(features.shape)
    # print(labels.shape)
    # print(features_append.shape)
    # print(labels_append.shape)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)

    # print('aaaaaaaa')
    # print(features.shape)
    # print(labels.shape)

    # print(adj)
    # print(idx_train)
    # print(idx_train.shape)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    new_data = data.clone()
    data.x = features
    data.y = labels

    # data.edge_index, _ = from_scipy_sparse_matrix(adj)  Doesn't work because of scipy instead of tensor

    data.edge_index = adj.indices()
    # row, col, edge_attr = adj.t().coo()  # Doesn't work because of scipy instead of tensor
    # data.edge_index = torch.stack([row, col], dim=0)

    # data.edge_index = adj.nonzero().t().contiguous()

    # print(data.x.shape)
    # data.train_mask = torch.tensor([False for _ in range(data.x.shape[0])])
    # print(data.train_mask.shape)
    # print(idx_train)
    # print(idx_train.shape)
    # data.train_mask[idx_train] = torch.tensor([True for _ in range(idx_train.shape[0])])
    
    data.train_mask = torch.cat((data.train_mask, torch.tensor([True for _ in range(data.x.shape[0] - data.train_mask.shape[0])], device=data.train_mask.device)), 0)
    data.val_mask = torch.cat((data.val_mask, torch.tensor([False for _ in range(data.x.shape[0] - data.val_mask.shape[0])], device=data.val_mask.device)), 0)
    data.test_mask = torch.cat((data.test_mask, torch.tensor([False for _ in range(data.x.shape[0] - data.test_mask.shape[0])], device=data.test_mask.device)), 0)

    return data


def src_imgagn(data, x_gen, y_gen, edge_index_gen):
    data_new = data.clone().attach()
    data_new.x = torch.cat((data.x, x_gen.detach()), 0)
    data_new.y = torch.cat((data.y, y_gen.detach()), 0)
    data_new.edge_index = torch.cat((data.edge_index, edge_index_gen.detach()), 0)
    return data
