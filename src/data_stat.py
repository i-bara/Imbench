import torch
import random
import numpy as np
import os
from data_utils import get_dataset, get_longtail_split
import json


seed = 100

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

output_path = '../data/'

if not os.path.isdir(output_path):
        os.mkdir(output_path)

for dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'squirrel', 'Actor', 'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv']:
    if not os.path.isdir(os.path.join(output_path, dataset_name)):
        os.mkdir(os.path.join(output_path, dataset_name))

    path = '../datasets/'
    path = os.path.join(path, dataset_name)
    dataset = get_dataset(dataset_name, path, split_type='full')
    data_ = dataset[0]
    n_cls = data_.y.max().item() + 1
    n_sample = data_.x.shape[0]
    n_feat = data_.x.shape[1]
    data = data_.to('cpu')

    masks = dict()

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'squirrel', 'Actor', 'Wisconsin', 'Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv']:
        for imb_ratio in [100, 20, 1]:
            masks['train'], masks['val'], masks['test'] = get_longtail_split(data, imb_ratio=imb_ratio, train_ratio=0.1, val_ratio=0.1)
            a = dict()
            for ds in ['train', 'val', 'test']:
                a[ds] = torch.arange(n_sample, dtype=torch.int32, device='cpu')[masks[ds]].tolist()
            with open(os.path.join(output_path, dataset_name, f'{imb_ratio}.json'), 'w+') as f:
                json.dump(a, f, indent=4)
    else:
        raise NotImplementedError
