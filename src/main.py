import os
import os.path as osp
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F

from args import parse_args
from data_utils import get_dataset, get_idx_info, make_longtailed_data_remove, get_step_split, lt, step, get_longtail_split
from gens import sampling_node_source, neighbor_sampling, neighbor_sampling_ens, duplicate_neighbor, saliency_mixup, saliency_mixup_ens, sampling_idx_individual_dst, MeanAggregation_ens, src_smote, src_imgagn
from nets import create_gcn, create_gat, create_sage, create_generator
from utils import CrossEntropy, euclidean_dist
from tam import adjust_output
from sklearn.metrics import balanced_accuracy_score, f1_score
from neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist




import warnings
warnings.filterwarnings("ignore")


# ens

def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_input[0].data


aggregator = MeanAggregation_ens()

# gen

def train_gen():
    global class_num_list, idx_info
    global data_train_mask, data_val_mask, data_test_mask
    model_gen.train()
    optimizer_gen.zero_grad()

    z = np.random.normal(0, 1, (n_gen, 100))
    z = torch.tensor(z, dtype=torch.float32, device=data.x.device)
    adj_min = model_gen(z)
    print(adj_min.shape)
    x_gen = torch.zeros((n_gen, data.x.shape[1]), dtype=data.x.dtype, device=data.x.device)
    edge_index_gen = torch.zeros((2, 0), dtype=data.edge_index.dtype, device=data.edge_index.device)

    stat_gen(data=data_new, data_train_mask=data_train_mask, data_val_mask=data_val_mask, data_test_mask=data_test_mask, data_gen_mask=data_gen_mask)

    for i in range(n_cls):
        w = F.softmax(adj_min[idx_info_gen[i] - n_ori, :][:, idx_info[i]], dim=1)
        x_gen[idx_info_gen[i] - n_ori] = torch.mm(w, data.x[idx_info[i]])
        edge_index_gen_ = torch.where(w > 1 / w.shape[1], 1., 0.).nonzero().t().contiguous()
        debug_all(edge_index_gen_)
        edge_index_gen_0 = idx_info_gen[i][edge_index_gen_[0]]
        edge_index_gen_1 = idx_info[i][edge_index_gen_[1]]
        edge_index_gen__ = torch.stack((edge_index_gen_0, edge_index_gen_1), dim=0)
        print(edge_index_gen.shape)
        print(edge_index_gen_0.shape)
        print(edge_index_gen_1.shape)
        print(edge_index_gen__.shape)
        edge_index_gen = torch.cat((edge_index_gen, edge_index_gen__), dim=1)

    data_new.x = torch.cat((data.x.clone().detach(), x_gen.clone().detach()), dim=0)
    data_new.edge_index = torch.cat((data.edge_index.clone().detach(), edge_index_gen.clone().detach()), dim=1)

    output, output_gen = model(data_new.x, data_new.edge_index)

    dist = 0
    for i in range(n_cls):
        x_cls = data_new.x[idx_info[i]]
        x_gen_cls = data_new.x[idx_info_gen[i]]
        dist += euclidean_dist(x_cls, x_gen_cls).mean()
    loss_gen = F.cross_entropy(output_gen[data_gen_mask], torch.LongTensor(n_gen).fill_(0).to(data_new.y.device)) \
             + F.cross_entropy(output[data_gen_mask], data_new.y[data_gen_mask]) \
             + dist

    loss_gen.backward()



    # with torch.no_grad(): # no need to val
    #     model.eval()
    #     output = model(data.x, data.edge_index[:,train_edge_mask])
        
    #     dist = 0
    #     for i in range(n_cls):
    #         x_cls = data_new.x[idx_info[i]]
    #         x_gen_cls = data_new.x[idx_info_gen[i]]
    #         dist += euclidean_dist(x_cls, x_gen_cls).mean()
    #     loss_gen = F.cross_entropy(output_gen[data_gen_mask], torch.LongTensor(n_gen).fill_(0)) \
    #             + F.cross_entropy(output[data_gen_mask], data_new.y[data_gen_mask]) \
    #             + dist

    optimizer_gen.step()
    # scheduler.step(val_loss_gen)
    return

# public

def train():
    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    model.train()
    optimizer.zero_grad()

    if args.method == 'sha':

        if epoch > args.warmup:
            
            # identifying source samples
            debug_shape(prev_out)
            debug_shape(train_idx)
            prev_out_local = prev_out[train_idx]
            sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, args.tau, args.max, args.no_mask) 
            
            # semimxup
            new_edge_index = neighbor_sampling(data.x.size(0), data.edge_index[:,train_edge_mask], sampling_src_idx, neighbor_dist_list)
            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index[:,train_edge_mask], sampling_src_idx)
            new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)

        output = model(new_x, new_edge_index)
        prev_out = (output[:data.x.size(0)]).detach().clone()
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device= data.x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim =0)
        _new_y = data.y[sampling_src_idx].clone()
        new_y = torch.cat((data.y[data_train_mask], _new_y),dim =0)
        criterion(output[new_train_mask], new_y).backward()

    elif args.method in ['ens', 'tam']:
        # Hook saliency map of input features
        model.conv1.temp_weight.register_backward_hook(backward_hook)

        # Sampling source and destination nodes
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        ori_saliency = saliency[:data.x.shape[0]] if (saliency != None) else None

        # Augment nodes
        if epoch > args.warmup:
            with torch.no_grad():
                prev_out = aggregator(prev_out, data.edge_index[:,train_edge_mask])
                prev_out = F.softmax(prev_out / args.tau, dim=1).detach().clone()
            new_edge_index, dist_kl = neighbor_sampling_ens(data.x.size(0), data.edge_index[:,train_edge_mask], sampling_src_idx, sampling_dst_idx,
                                        neighbor_dist_list, prev_out, train_node_mask)
            new_x = saliency_mixup_ens(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=args.keep_prob)
        else:
            new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index[:,train_edge_mask], sampling_src_idx)
            dist_kl, ori_saliency = None, None
            new_x = saliency_mixup_ens(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl)
        new_x.requires_grad = True

        # Get predictions
        output = model(new_x, new_edge_index, None)
        prev_out = (output[:data.x.size(0)]).detach().clone() # logit propagation

        ## Train_mask modification ##
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device= data.x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim =0)

        if args.method == 'tam':
            ## Label modification ##
            _new_y = data.y[sampling_src_idx].clone()
            new_y = torch.cat((data.y[data_train_mask], _new_y),dim =0)

            ## Apply TAM ##
            output = adjust_output(args, output, new_edge_index, torch.cat((data.y,_new_y),dim =0), \
                new_train_mask, aggregator, class_num_list, epoch)

            ## Compute Loss ##
            criterion(output, new_y).backward()
        else:
            ## Label modification ##
            new_y = data.y[sampling_src_idx].clone()
            new_y = torch.cat((data.y[data_train_mask], new_y),dim =0)

            ## Compute Loss ##
            criterion(output[new_train_mask], new_y).backward()

    elif args.method == 'imgagn':
        # if epoch_gen == 8 and epoch == 40:
        #     debug_all(data_new.x)
        output, output_gen = model(data_new.x.detach(), data_new.edge_index.detach())
        loss = F.cross_entropy(output_gen[data_train_mask | data_gen_mask], torch.cat((torch.LongTensor(n_train).fill_(0), torch.LongTensor(n_gen).fill_(1))).to(data_new.y.device)) \
             + F.cross_entropy(output[data_train_mask | data_gen_mask], data_new.y[data_train_mask | data_gen_mask])
                    # +loss_dis
        loss.backward()

    else: ## Vanilla Train ##
        output = model(data.x, data.edge_index[:,train_edge_mask], None)
        criterion(output[data_train_mask], data.y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        if args.method == 'imgagn':
            output, output_gen = model(data_new.x, data_new.edge_index)
            val_loss = F.cross_entropy(output_gen[data_val_mask], torch.LongTensor(n_val).fill_(0).to(data_new.y.device)) \
                     + F.cross_entropy(output[data_val_mask], data_new.y[data_val_mask])
        else:
            output = model(data.x, data.edge_index[:,train_edge_mask])
            val_loss = F.cross_entropy(output[data_val_mask], data.y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss)
    return

@torch.no_grad()
def test():
    model.eval()
    if args.method == 'imgagn':
        logits, _ = model(data.x, data.edge_index[:,train_edge_mask])
    else:
        logits = model(data.x, data.edge_index[:,train_edge_mask])
    accs, baccs, f1s = [], [], []
    for mask in [data_train_mask, data_val_mask, data_test_mask]:
        if args.method == 'imgagn':
            mask = mask[torch.logical_not(data_gen_mask)]
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
    return accs, baccs, f1s

args = parse_args()


def debug(*args_):
    if args.debug:
        print(*args_)


def debug_shape(tensor):
    if args.debug:
        print(tensor.shape)


def debug_all(*args_):
    if args.debug:
        # torch.set_printoptions(threshold=10_000)
        torch.set_printoptions(profile="full")
        print(*args_)
        torch.set_printoptions(profile="default")


# if args.method not in ['vanilla', 'ens', 'sha']:
#     smote()
#     exit()

# if args.method == 'smote':
#     args.epoch = 1200  # smote needs more epoches to converge

seed = args.seed
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(args.device)

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

debug(f'seed={seed}')

path = args.data_path
path = osp.join(path, args.dataset)
dataset = get_dataset(args.dataset, path, split_type='full')
data = dataset[0]
n_cls = data.y.max().item() + 1
data = data.to(device)

debug(args.dataset)

# debug_all(data.y)



# data_train_mask along with train_idx
# data_val_mask along with train_idx
# data_test_mask along with train_idx

# train_node_mask = data_train_mask | data_val_mask | data_test_mask
# train_edge_mask = edge for train_node_mask(Planetoid) or all true(Amazon)

# Set the mask for the dataset by 1:1:8

# if args.dataset == 'ogbn-arxiv':
#     data.y = data.y[:, 0]
#     train_num = int(data.y.shape[0] * 0.1)
#     val_num = int(data.y.shape[0] * 0.2)
#     test_num = data.y.shape[0]
#     idx_train = torch.arange(train_num, device=data.y.device)
#     idx_val = torch.arange(train_num, val_num, device=data.y.device)
#     idx_test = torch.arange(val_num, test_num, device=data.y.device)
# else:


def stat(data, data_train_mask, data_val_mask, data_test_mask):
    idx_train = torch.tensor(range(data_train_mask.shape[0]), device=data_train_mask.device)[data_train_mask]
    idx_val = torch.tensor(range(data_val_mask.shape[0]), device=data_val_mask.device)[data_val_mask]
    idx_test = torch.tensor(range(data_test_mask.shape[0]), device=data_test_mask.device)[data_test_mask]

    # Output the split distribution
    debug('class   train   val     test    total   ')
    for i in range(data.y.max().item() + 1):
        idx_train_i = idx_train[(data.y == i)[idx_train]]
        idx_val_i = idx_val[(data.y == i)[idx_val]]
        idx_test_i = idx_test[(data.y == i)[idx_test]]
        debug('%-4d    %-8d%-8d%-8d%-8d' % (i, idx_train_i.shape[0], idx_val_i.shape[0], idx_test_i.shape[0], idx_train_i.shape[0] + idx_val_i.shape[0] + idx_test_i.shape[0]))
    debug('total   %-8d%-8d%-8d%-8d' % (idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], idx_train.shape[0] + idx_val.shape[0] + idx_test.shape[0]))


def stat_gen(data, data_train_mask, data_val_mask, data_test_mask, data_gen_mask):
    idx_train = torch.tensor(range(data_train_mask.shape[0]), device=data_train_mask.device)[data_train_mask]
    idx_val = torch.tensor(range(data_val_mask.shape[0]), device=data_val_mask.device)[data_val_mask]
    idx_test = torch.tensor(range(data_test_mask.shape[0]), device=data_test_mask.device)[data_test_mask]
    idx_gen = torch.tensor(range(data_gen_mask.shape[0]), device=data_gen_mask.device)[data_gen_mask]

    # Output the split distribution
    debug('class   train   val     test    gen     total   ')
    for i in range(data.y.max().item() + 1):
        idx_train_i = idx_train[(data.y == i)[idx_train]]
        idx_val_i = idx_val[(data.y == i)[idx_val]]
        idx_test_i = idx_test[(data.y == i)[idx_test]]
        idx_gen_i = idx_gen[(data.y == i)[idx_gen]]
        debug('%-4d    %-8d%-8d%-8d%-8d%-8d' % (i, idx_train_i.shape[0], idx_val_i.shape[0], idx_test_i.shape[0], idx_gen_i.shape[0], idx_train_i.shape[0] + idx_val_i.shape[0] + idx_test_i.shape[0] + idx_gen_i.shape[0]))
    debug('total   %-8d%-8d%-8d%-8d%-8d' % (idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], idx_gen.shape[0], idx_train.shape[0] + idx_val.shape[0] + idx_test.shape[0] + idx_gen.shape[0]))


if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'squirrel', 'Actor', 'Wisconsin']:
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data_train_mask = data.train_mask.clone()
        data_val_mask = data.val_mask.clone()
        data_test_mask = data.test_mask.clone()
    else:
        data_train_mask = data.train_mask[:, 0].clone()  # chameleon and squirrel dataset provides 10 masks
        data_val_mask = data.val_mask[:, 0].clone()
        data_test_mask = data.test_mask[:, 0].clone()

    data_train_mask, data_val_mask, data_test_mask = get_longtail_split(data, imb_ratio=args.imb_ratio, train_ratio=0.1, val_ratio=0.1)

    debug(f'feature size: {data.x.shape[1]}')
    debug(f'number of edges: {data.edge_index.shape[1]}')
    stat(data, data_train_mask, data_val_mask, data_test_mask)

    train_node_mask = data_train_mask | data_val_mask | data_test_mask
    train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

    idx_info = [torch.arange(data.y.shape[0], device=data.y.device)[(data.y == i) & data_train_mask] for i in range(n_cls)]
    class_num_list = [idx_info[i].shape[0] for i in range(n_cls)]

    # stat(data, data_train_mask, data_val_mask, data_test_mask)
    # data_train_mask, train_node_mask, train_edge_mask, class_num_list, idx_info = lt(data=data, data_train_mask=data_train_mask, imb_ratio=args.imb_ratio)
    # stat(data, data_train_mask, data_val_mask, data_test_mask)

    assert torch.all(train_node_mask == data_train_mask | data_val_mask | data_test_mask)
    for i in range(train_edge_mask.shape[0]):
        row, col = data.edge_index[0][i], data.edge_index[1][i]
        if train_edge_mask[i]:  # edge in mask iff both nodes in mask
            assert train_node_mask[row] and train_node_mask[col]
        else:
            assert not train_node_mask[row] or not train_node_mask[col]

    # print(class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask) 

elif args.dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv']:
    if args.dataset == 'ogbn-arxiv':
        data.y = data.y[:, 0]

    data_train_mask, data_val_mask, data_test_mask, class_num_list, idx_info = step(data=data, imb_ratio=args.imb_ratio)

    debug(f'feature size: {data.x.shape[1]}')
    debug(f'number of edges: {data.edge_index.shape[1]}')
    stat(data, data_train_mask, data_val_mask, data_test_mask)

    train_node_mask = data_train_mask | data_val_mask | data_test_mask
    train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
else:
    raise NotImplementedError

for i in range(n_cls):
    assert torch.all(torch.arange(data.y.shape[0], device=data.y.device)[(data.y == i) & data_train_mask] == idx_info[i])
    assert idx_info[i].shape[0] == class_num_list[i]


if args.method == 'smote':
    data = src_smote(data)
    data_train_mask = torch.cat((data_train_mask, torch.ones(data.x.shape[0] - data_train_mask.shape[0], dtype=torch.bool, device=data_train_mask.device)), 0)
    data_val_mask = torch.cat((data_val_mask, torch.zeros(data.x.shape[0] - data_val_mask.shape[0], dtype=torch.bool, device=data_val_mask.device)), 0)
    data_test_mask = torch.cat((data_test_mask, torch.zeros(data.x.shape[0] - data_test_mask.shape[0], dtype=torch.bool, device=data_test_mask.device)), 0)
    train_edge_mask = torch.cat((train_edge_mask, torch.ones(data.edge_index.shape[1] - train_edge_mask.shape[0], dtype=torch.bool, device=train_edge_mask.device)), 0)




train_idx = data_train_mask.nonzero().squeeze()
val_idx = data_val_mask.nonzero().squeeze()  # not used yet
test_idx = data_test_mask.nonzero().squeeze()  # not used yet

labels_local = data.y.view([-1])[train_idx]
train_idx_list = train_idx.cpu().tolist()
local2global = {i:train_idx_list[i] for i in range(len(train_idx_list))}
global2local = dict([val, key] for key, val in local2global.items())
idx_info_list = [item.cpu().tolist() for item in idx_info]
idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]


tensor_path = osp.join(path, 'tensors')
debug(tensor_path)

# if False:
    # pass
# tensor_path = osp.join(path, 'tensors')
# if not osp.isdir(tensor_path):
#     os.system(f'mkdir {tensor_path}')
# neighbor_name = f'neighbor_{args.dataset}_{args.gdc}.pt'
# neighbor_path = osp.join(tensor_path, neighbor_name)
# if osp.isfile(neighbor_path):
#     neighbor_dist_list = torch.load(neighbor_path)
# else:
def neighbor(data, train_edge_mask):
    if args.gdc=='ppr':
        neighbor_dist_list = get_PPR_adj(data.x, data.edge_index[:,train_edge_mask], alpha=0.05, k=128, eps=None)
    elif args.gdc=='hk':
        neighbor_dist_list = get_heat_adj(data.x, data.edge_index[:,train_edge_mask], t=5.0, k=None, eps=0.0001)
    elif args.gdc=='none':
        neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index[:,train_edge_mask], data_train_mask, device)
    # torch.save(neighbor_dist_list, neighbor_path)
    return neighbor_dist_list


neighbor_dist_list = neighbor(data=data, train_edge_mask=train_edge_mask)

if args.net == 'GCN':
    model = create_gcn(nfeat=dataset.num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.n_layer, has_discriminator=args.method == 'imgagn')
elif args.net == 'GAT':
    model = create_gat(nfeat=dataset.num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.n_layer)
elif args.net == "SAGE":
    model = create_sage(nfeat=dataset.num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.n_layer)
else:
    raise NotImplementedError("Not Implemented Architecture!")

model = model.to(device)
criterion = CrossEntropy().to(device)

optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=args.weight_decay), dict(params=model.non_reg_params, weight_decay=0),], lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)

best_val_acc_f1 = 0
saliency, prev_out = None, None

if args.method == 'imgagn':
    ratio_generated = 1.
    class_num_max = max(class_num_list)
    class_num_gen_list = list(map(lambda x: round(ratio_generated * class_num_max - x), class_num_list))

    n_ori = data.x.shape[0]
    n_train = sum(class_num_list)
    n_val = sum(data_val_mask).item()
    n_gen = sum(class_num_gen_list)
    data_train_mask = torch.cat((data_train_mask, torch.zeros(n_gen, dtype=torch.bool, device=data.y.device)), dim=0)
    data_val_mask = torch.cat((data_val_mask, torch.zeros(n_gen, dtype=torch.bool, device=data.y.device)), dim=0)
    data_test_mask = torch.cat((data_test_mask, torch.zeros(n_gen, dtype=torch.bool, device=data.y.device)), dim=0)
    data_gen_mask = torch.cat((torch.zeros(n_ori, dtype=torch.bool, device=data.y.device), \
                              torch.ones(n_gen, dtype=torch.bool, device=data.y.device)), dim=0)
    
    train_idx = data_train_mask.nonzero().squeeze()
    val_idx = data_val_mask.nonzero().squeeze()  # not used yet
    test_idx = data_test_mask.nonzero().squeeze()  # not used yet
    gen_idx = data_test_mask.nonzero().squeeze()

    y_gen = []
    for i in range(n_cls):
        y_gen += [i] * class_num_gen_list[i]
    y_gen = torch.tensor(y_gen, dtype=data.y.dtype, device=data.y.device)

    data_new = data.clone().detach()
    data_new.y = torch.cat((data.y.clone().detach(), y_gen.clone().detach()), dim=0)
    idx_info_val = [torch.arange(data_new.y.shape[0], device=data_new.y.device)[(data_new.y == i) & data_val_mask] for i in range(n_cls)]
    idx_info_test = [torch.arange(data_new.y.shape[0], device=data_new.y.device)[(data_new.y == i) & data_test_mask] for i in range(n_cls)]
    idx_info_gen = [torch.arange(data_new.y.shape[0], device=data_new.y.device)[(data_new.y == i) & data_gen_mask] for i in range(n_cls)]

    model_gen = create_generator(n_ori)
    model_gen = model_gen.to(device)
    optimizer_gen = torch.optim.Adam([dict(params=model_gen.reg_params, weight_decay=args.weight_decay), dict(params=model.non_reg_params, weight_decay=0),], lr=args.lr)

    for epoch_gen in range(10):
        train_gen()
        for epoch in tqdm.tqdm(range(args.epoch // 10)):
            train()
            accs, baccs, f1s = test()
            train_acc, val_acc, tmp_test_acc = accs
            train_f1, val_f1, tmp_test_f1 = f1s
            val_acc_f1 = (val_acc + val_f1) / 2.
            if val_acc_f1 > best_val_acc_f1:
                best_val_acc_f1 = val_acc_f1
                test_acc = accs[2]
                test_bacc = baccs[2]
                test_f1 = f1s[2]
else:
    for epoch in tqdm.tqdm(range(args.epoch)):
        train()
        accs, baccs, f1s = test()
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, val_f1, tmp_test_f1 = f1s
        val_acc_f1 = (val_acc + val_f1) / 2.
        if val_acc_f1 > best_val_acc_f1:
            best_val_acc_f1 = val_acc_f1
            test_acc = accs[2]
            test_bacc = baccs[2]
            test_f1 = f1s[2]

print('acc: {:.9f}, bacc: {:.9f}, f1: {:.9f}'.format(test_acc*100, test_bacc*100, test_f1*100))
