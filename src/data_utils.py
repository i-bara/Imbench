from typing import List, Tuple, Dict, Optional, Union, Any, Callable, Iterable, Sequence, Set, Mapping
import torch
import numpy as np
import random
from torch_scatter import scatter_add
from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(name, path, split_type='public'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == "chameleon" or name == "squirrel":
        from dataset.WikipediaNetwork import WikipediaNetwork
        dataset = WikipediaNetwork(path, name, transform = T.NormalizeFeatures())
    elif name == "Actor":
        from torch_geometric.datasets import Actor
        dataset = Actor(path, transform=T.NormalizeFeatures())
    elif name == "Wisconsin":
        from dataset.WebKB import WebKB
        dataset = WebKB(path, name, transform = T.NormalizeFeatures())
    # elif name == "chameleon" or name == "squirrel":
    #     from torch_geometric.datasets import WikipediaNetwork
    #     dataset = WikipediaNetwork(path, name, transform = T.NormalizeFeatures())
    # elif name == "Wisconsin":
    #     from torch_geometric.datasets import WebKB
    #     dataset = WebKB(path, name, transform = T.NormalizeFeatures())
    elif name == 'Amazon-Computers':
        from torch_geometric.datasets import Amazon
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    elif name == 'Amazon-Photo':
        from torch_geometric.datasets import Amazon
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    elif name == 'Coauthor-CS':
        from torch_geometric.datasets import Coauthor
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    elif name.startswith('ogbn'):
        data = PygNodePropPredDataset(name=name, root=path)
        data[0].y.squeeze_()
        return data
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset

def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label), device=label.device)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


def sort_(class_num_list):
    n_cls = len(class_num_list)

    class_num_list_tensor = torch.tensor(class_num_list)
    class_num_list_sorted_tensor, indices = torch.sort(class_num_list_tensor, descending=True)
    class_num_list_sorted = class_num_list_sorted_tensor.tolist()
    inv_indices = torch.zeros(n_cls, dtype=indices.dtype, device=indices.device)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i

    # assert class_num_list_sorted == class_num_list[indices]
    # assert class_num_list == class_num_list_sorted[inv_indices]
    assert torch.equal(class_num_list_sorted_tensor, class_num_list_tensor[indices])
    assert torch.equal(class_num_list_tensor, class_num_list_sorted_tensor[inv_indices])
    return class_num_list_tensor, indices, inv_indices


def sort(data, data_mask=None):
    if data_mask is None:
        y = data.y
    else:
        y = data.y[data_mask]
    
    n_cls = data.y.max().item() + 1

    class_num_list = []
    for i in range(n_cls):
        class_num_list.append(int((y == i).sum().item()))
    
    class_num_list_tensor = torch.tensor(class_num_list)
    class_num_list_sorted_tensor, indices = torch.sort(class_num_list_tensor, descending=True)
    class_num_list_sorted = class_num_list_sorted_tensor.tolist()
    inv_indices = torch.zeros(n_cls, dtype=indices.dtype, device=indices.device)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i

    # assert class_num_list_sorted == class_num_list[indices]
    # assert class_num_list == class_num_list_sorted[inv_indices]
    assert torch.equal(class_num_list_sorted_tensor, class_num_list_tensor[indices])
    assert torch.equal(class_num_list_tensor, class_num_list_sorted_tensor[inv_indices])
    return class_num_list_tensor, indices, inv_indices


def split_lt(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
    class_num_list = class_num_list[indices]  # sort
    mu = np.power(imb_ratio, 1 / (n_cls - 1))
    _mu = 1 / mu
    if imb_ratio == 1:
        n_max = n / n_cls
    else:
        n_max = n / (imb_ratio * mu - 1) * (mu - 1) * imb_ratio
    class_num_list_lt = []
    for i in range(n_cls):
        class_num_list_lt.append(round(min(max(n_max * np.power(_mu, i), 1), class_num_list[i].item() - keep)))
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt[inv_indices]  # unsort


def split_step(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
    class_num_list = class_num_list[indices]  # sort
    n_h = n_cls // 2
    n_t = n_cls - n_h
    t = n / (n_h * imb_ratio + n_t)
    h = t * imb_ratio
    class_num_list_lt = []
    for i in range(n_cls):
        if i < n_h:
            class_num_list_lt.append(min(int(h), class_num_list[i].item() - keep))
        else:
            class_num_list_lt.append(min(int(t), class_num_list[i].item() - keep))
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt[inv_indices]  # unsort


def split_same(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
    class_num_list_lt = []
    for i in range(n_cls):
        class_num_list_lt.append(min(round(n / n_cls), class_num_list[i].item() - keep))
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt


def split_natural(
    class_num_list: List[int], 
    indices: torch.LongTensor, 
    inv_indices: torch.LongTensor, 
    imb_ratio: float, 
    n_cls: int, 
    n: float,
) -> torch.LongTensor:
    r"""
    Split data into two parts with the natural number of nodes in each class.
    
    Args:
        class_num_list (List[int]): Number of nodes in each class.
        indices (torch.LongTensor): Indices of classes.
        inv_indices (torch.LongTensor): Inverse indices of classes.
        imb_ratio (float): Imbalance ratio.
        n_cls (int): Number of classes.
        n (float): Splitted number of nodes.
    
    Returns:
        torch.LongTensor: Splitted number of nodes in each class.
    """
    class_num_list_lt = [min(round(n / n_cls), class_num_list[c].item()) for c in range(n_cls)]
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt


def choose(class_num_list, class_num_list_lt, indices, data, data_mask, choose_deg='smallest', keep=0):
    node_mask = data_mask.clone().detach()
    for i in indices:
        if choose_deg is not None:
            idx = torch.arange(data.y.shape[0], dtype=torch.int64, device=data.y.device)[(data.y == i)]
            remove = class_num_list[i] - class_num_list_lt[i]
            for r in range(10):
                # Remove connection with removed nodes
                row, col = data.edge_index[0], data.edge_index[1]
                row_mask = node_mask[row]
                col_mask = node_mask[col]
                edge_mask = row_mask & col_mask

                # Compute degree
                degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=data.y.shape[0]).to(data.y.device)
                degree = degree[idx]

                # Remove nodes with low degree first (number increases as round increases)
                # Accumulation does not be problem since
                _, remove_idx = torch.topk(degree, ((r + 1) * remove) // 10, largest=choose_deg == 'largest')  # GrphaSHA uses largest=True but we have forbidden here
                remove_idx = idx[remove_idx]
                node_mask[remove_idx] = False
        else:
            idx = torch.arange(data.y.shape[0], dtype=torch.int64, device=data.y.device)[node_mask & (data.y == i)]
            node_mask[idx[min(len(idx) - keep, class_num_list_lt[i]):]] = False
    
    assert torch.equal(node_mask & data_mask, node_mask)

    return node_mask


def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        # 4.18: It may cause small dataset failed, set at least 1
        class_num_list.append(int(min(max(sorted_n_data[0].item() * np.power(mu, i), 1), sorted_n_data[i])))
        # assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        # class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1: # We does not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask), device=label.device)
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.cpu().numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    # assert set(node_mask) <= set(train_mask)
    assert torch.all(node_mask[torch.logical_not(train_mask)])
    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask

def get_step_split2(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass):
    base_valid_each = valid_each

    head_list = [i for i in range(nclass//2)] 

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    if imb_ratio == 0:
        base_train_each = -1
    else:
        base_train_each = int( len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        if base_train_each == -1:
            train_list = [0 for _ in range(nclass)]
            for iter0 in all_idx:
                iter_label = all_label[iter0]
                train_list[iter_label]+=1
            idx2train[i_h] = int(train_list[i_h] * labeling_ratio)
        else:
            idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1) 

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list: 
        if base_train_each == -1:
            train_list = [0 for _ in range(nclass)]
            for iter0 in all_idx:
                iter_label = all_label[iter0]
                train_list[iter_label]+=1
            idx2train[i_t] = int(train_list[i_t] * labeling_ratio)
        else:
            idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx, valid_idx, test_idx, train_node


def get_longtail_split(data, imb_ratio, train_ratio, val_ratio):
    class_num_list, indices, inv_indices = sort(data=data)

    # train
    class_num_list_train = split_lt(class_num_list=class_num_list, indices=indices, inv_indices=inv_indices, \
                          imb_ratio=imb_ratio, n_cls=data.y.max().item() + 1, n=data.y.shape[0] * train_ratio)

    data_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=data.y.device)

    data_train_mask = choose(class_num_list, class_num_list_train, indices, data, data_mask, choose_deg=None, keep=20)  # At least 20 for val and test

    # val
    class_num_list_val = split_same(class_num_list=class_num_list, indices=indices, inv_indices=inv_indices, \
                          imb_ratio=imb_ratio, n_cls=data.y.max().item() + 1, n=data.y.shape[0] * val_ratio)

    data_mask = data_mask & torch.logical_not(data_train_mask)

    data_val_mask = choose(class_num_list, class_num_list_val, indices, data, data_mask, choose_deg=None, keep=10)  # At least 10 for test

    # test
    data_test_mask = data_mask & torch.logical_not(data_val_mask)

    return data_train_mask, data_val_mask, data_test_mask


def get_step_split(data, imb_ratio, train_ratio, val_ratio):
    class_num_list, indices, inv_indices = sort(data=data)

    # train
    class_num_list_train = split_step(class_num_list=class_num_list, indices=indices, inv_indices=inv_indices, \
                          imb_ratio=imb_ratio, n_cls=data.y.max().item() + 1, n=data.y.shape[0] * train_ratio)

    data_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=data.y.device)

    data_train_mask = choose(class_num_list, class_num_list_train, indices, data, data_mask, choose_deg=None, keep=20)  # At least 20 for val and test

    # val
    class_num_list_val = split_same(class_num_list=class_num_list, indices=indices, inv_indices=inv_indices, \
                          imb_ratio=imb_ratio, n_cls=data.y.max().item() + 1, n=data.y.shape[0] * val_ratio)

    data_mask = data_mask & torch.logical_not(data_train_mask)

    data_val_mask = choose(class_num_list, class_num_list_val, indices, data, data_mask, choose_deg=None, keep=10)  # At least 10 for test

    # test
    data_test_mask = data_mask & torch.logical_not(data_val_mask)

    return data_train_mask, data_val_mask, data_test_mask


def get_natural_split(data, imb_ratio, train_ratio, val_ratio):
    class_num_list, indices, inv_indices = sort(data=data)

    # train
    class_num_list_test = split_same(class_num_list=class_num_list, indices=indices, inv_indices=inv_indices, \
                          imb_ratio=imb_ratio, n_cls=data.y.max().item() + 1, n=data.y.shape[0] * (1 - train_ratio - val_ratio))

    data_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=data.y.device)

    data_test_mask = choose(class_num_list, class_num_list_test, indices, data, data_mask, choose_deg=None, keep=20)  # At least 20 for val and test

    # val
    class_num_list_val = split_same(class_num_list=class_num_list, indices=indices, inv_indices=inv_indices, \
                          imb_ratio=imb_ratio, n_cls=data.y.max().item() + 1, n=data.y.shape[0] * val_ratio)

    data_mask = data_mask & torch.logical_not(data_test_mask)

    data_val_mask = choose(class_num_list, class_num_list_val, indices, data, data_mask, choose_deg=None, keep=10)  # At least 10 for test

    # test
    data_train_mask = data_mask & torch.logical_not(data_val_mask)

    return data_train_mask, data_val_mask, data_test_mask


def lt(data, data_train_mask, imb_ratio):
    n_cls = data.y.max().item() + 1

    ## Data statistic ##
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data

    ## Construct a long-tailed graph ##
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
        make_longtailed_data_remove(data.edge_index, data.y, n_data, n_cls, imb_ratio, data_train_mask.clone())

    return data_train_mask, train_node_mask, train_edge_mask, class_num_list, idx_info


def step(data, imb_ratio):
    n_cls = data.y.max().item() + 1

    train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=imb_ratio, \
                                                                valid_each=int(data.x.shape[0] * 0.1 / n_cls), \
                                                                labeling_ratio=0.1, \
                                                                all_idx=[i for i in range(data.x.shape[0])], \
                                                                all_label=data.y.cpu().detach().numpy(), \
                                                                nclass=n_cls)

    data_train_mask = torch.zeros(data.x.shape[0]).bool().to(data.y.device)
    data_val_mask = torch.zeros(data.x.shape[0]).bool().to(data.y.device)
    data_test_mask = torch.zeros(data.x.shape[0]).bool().to(data.y.device)
    data_train_mask[train_idx] = True
    data_val_mask[valid_idx] = True
    data_test_mask[test_idx] = True

    class_num_list = [len(item) for item in train_node]
    idx_info = [torch.tensor(item) for item in train_node]

    return data_train_mask, data_val_mask, data_test_mask, class_num_list, idx_info


def separator_ht(dist, head_ratio=0.4): # Head / Tail separator
    class_num_list_tensor, indices, inv_indices = sort_(class_num_list=dist)

    # head = head_ratio
    # tail = 1 - head_ratio
    head_idx = int(len(dist) * head_ratio)
    ht_dict = {}

    if head_idx == 0:
        ht_dict['H'] = indices[torch.arange(1)]
        ht_dict['T'] = indices[torch.arange(1, dist)]
        return ht_dict

    else:
        ht_dict['H'] = indices[torch.arange(head_idx)]
        ht_dict['T'] = indices[torch.arange(head_idx, len(dist))]
        return ht_dict
        # crierion = dist[indices[head_idx]]

        # case1_h = sum(dist >= crierion)
        # case1_t = sum(dist < crierion)

        # case2_h = sum(dist > crierion)
        # case2_t = sum(dist <= crierion)

        # gap_case1 = abs(case1_h/case1_t - head/tail)
        # gap_case2 = abs(case2_h/case2_t - head/tail)

        # if gap_case1 < gap_case2:
        #     idx = sum(dist >= crierion)
        #     ht_dict['H'] = indices[torch.arange(idx)]
        #     ht_dict['T'] = indices[torch.arange(idx, len(dist))]

        # elif gap_case1 > gap_case2:
        #     idx = sum(dist > crierion)
        #     ht_dict['H'] = indices[torch.arange(idx)]
        #     ht_dict['T'] = indices[torch.arange(idx, len(dist))]

        # else:
        #     rand = random.choice([1, 2])
        #     if rand == 1:
        #         idx = sum(dist >= crierion)
        #         ht_dict['H'] = indices[torch.arange(idx)]
        #         ht_dict['T'] = indices[torch.arange(idx, len(dist))]
        #     else:
        #         idx = sum(dist > crierion)
        #         ht_dict['H'] = indices[torch.arange(idx)]
        #         ht_dict['T'] = indices[torch.arange(idx, len(dist))]

        # return ht_dict


def degree(data):
    return torch.tensor([(data.edge_index == i).sum().item() for i in range(data.y.shape[0])], dtype=torch.float32, device=data.y.device)


def separate_class_degree(adj, idx_train_set_class, below=None):
    idx_train_set = {}
    idx_train_set['HH'] = []
    idx_train_set['HT'] = []
    idx_train_set['TH'] = []
    idx_train_set['TT'] = []

    adj_dense = adj.to_dense()
    adj_dense[adj_dense != 0] = 1
    degrees = np.array(list(map(int, torch.sum(adj_dense, dim=0))))

    for sep in ['H', 'T']:
        if len(idx_train_set_class[sep]) == 0:
            continue

        elif len(idx_train_set_class[sep]) == 1:
            idx = idx_train_set_class[sep]
            if sep == 'H':
                rand = random.choice(['HH', 'HT'])
                idx_train_set[rand].append(int(idx))
            elif sep == 'T':
                rand = random.choice(['TH', 'TT'])
                idx_train_set[rand].append(int(idx))

        else:
            degrees_idx_train = degrees[idx_train_set_class[sep]]
            gap_head = abs(degrees_idx_train - (below+1))
            gap_tail = abs(degrees_idx_train - below)

            if sep == 'H':
                idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                if sum(gap_head == gap_tail) > 0:
                    for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                        rand = random.choice(['HH', 'HT'])
                        idx_train_set[rand].append(int(idx))

            elif sep == 'T':
                idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                if sum(gap_head == gap_tail) > 0:
                    for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                        rand = random.choice(['TH', 'TT'])
                        idx_train_set[rand].append(int(idx))

    for idx in ['HH', 'HT', 'TH', 'TT']:
        random.shuffle(idx_train_set[idx])
        idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

    return idx_train_set, degrees


def adj_mse_loss(adj_rec, adj_tgt):
    
    adj_tgt[adj_tgt != 0] = 1

    edge_num = adj_tgt.nonzero().shape[0] #number of non-zero
    total_num = adj_tgt.shape[0]**2 #possible edge

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2) # element-wise

    return loss


def adj(x, edge_index):
    adj = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float32, device=edge_index.device)
    for i, j in zip(edge_index[0], edge_index[1]):
        adj[i, j] = 1.
    return adj


def idx(mask, n_cls):
    return torch.arange(n_cls, device=mask.device)[mask]
            

def inv(idx, n_cls):
    inv_idx = torch.zeros(n_cls, dtype=idx.dtype, device=idx.device)
    for i in range(n_cls):
        inv_idx[idx[i].item()] = i
    return inv_idx


def scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1/2) * np.cos((epoch*np.pi) / curriculum_ep) + 1/2
