import argparse
from baselines.baseline import Baseline
from baselines.gnn import gnn
from baselines.lte4g import lte4g
from baselines.renode import renode
from baselines.topoauc import topoauc
from baselines.hyperimba import hyperimba
from baselines.pastel import pastel
from baselines.mixup import mixup
from baselines.dpgnn import dpgnn
from baselines.ens import ens
from baselines.sha import sha
from baselines.tam import tam
from baselines.igraphmix import igraphmix
from baselines.upsample import upsample
from baselines.mix import mix
from baselines.mixv3 import mix as mixv2
from baselines.bat import Bat
from baselines.graphmix import GraphMix
from baselines.mmixup import Mmixup
from baselines.reweight import Reweight


baseline_dict = {
    'vanilla': gnn,
    'lte4g': lte4g,
    'renode': renode,
    'topoauc': topoauc,
    'hyperimba': hyperimba,
    'pastel': pastel,
    'mixup': mixup,
    'dpgnn': dpgnn,
    'ens': ens,
    'sha': sha,
    'tam': tam,
    'igraphmix': igraphmix,
    'upsample': upsample,
    'mix_old': mix,
    'mix': mixv2,
    'bat': Bat,
    'graphmix': GraphMix,
    'mmixup': Mmixup,
    'reweight': Reweight,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument('--method', type=str, choices=[
        'vanilla', 'drgcn', 'smote', 'imgagn', 'lte4g', 'dpgnn', 'mixup', 
        'ens', 'tam', 'topoauc', 'sha', 'renode', 'pastel', 'hyperimba', 
        'igraphmix', 'upsample', 'mix_old', 'mix', 'bat', 'graphmix', 'mmixup', 'reweight'], 
        default='mix', 
        help='the method used to train')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    # Debug
    parser.add_argument('--debug', action="store_true", help='whether to debug')

    # Storage
    parser.add_argument('--output', type=str, help='path to store output')

    # Seed
    parser.add_argument('--seed', type=int, default=100, help='seed')

    # Data
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'squirrel', 'Actor', 'Wisconsin', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS', 'ogbn-products', 'ogbn-proteins', 'ogbn-arxiv', 'ogbn-papers100M', 'ogbn-mag'], default='Cora', help='dataset name')
    parser.add_argument('--data_path', type=str, default='datasets/', help='data path')
    parser.add_argument('--split', type=str, choices=['lt', 'st', 'nt', 'long-tailed', 'step-tailed', 'natural'], default='lt', help='split method, must be one of [long-tailed (lt), step (st), natural (nt)]')
    parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')

    # Backbone
    parser.add_argument('--backbone', type=str, choices=['GCN', 'GAT', 'SAGE'], default='GCN', help='GNN bachbone')
    parser.add_argument('--n_layer', type=int, default=2, help='the number of layers')
    parser.add_argument('--feat_dim', type=int, default=256, help='feature dimension')
    # GAT
    parser.add_argument('--n_head', type=int, default=8, help='the number of heads in GAT')

    # Hyperparameter
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    
    
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args, _ = parser.parse_known_args()
    baseline = None
    if args.method in baseline_dict:
        def parse(parsing):
            try:
                parsing.parse_args(parser)
            except AttributeError:
                return
            for parsing_base in parsing.__bases__:
                parse(parsing_base)
        
        print(f'method: {args.method}')
        baseline = baseline_dict[args.method]
        parse(baseline)
        args = parser.parse_args()
        print(args)
        return args, baseline
    
    parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
    parser.add_argument('--keep_prob', type=float, default=0.01, help='keeping probability') # used in ens
    parser.add_argument('--tau', type=int, default=2, help='temperature in the softmax function when calculating confidence-based node hardness') # used in sha, and ens, tam as pred_tau
    parser.add_argument('--tam_alpha', type=float, default=2.5, help='coefficient of ACM') # used in tam
    parser.add_argument('--tam_beta', type=float, default=0.5, help='coefficient of ADM') # used in tam
    parser.add_argument('--temp_phi', type=float, default=1.2, help='classwise temperature') # used in tam

    parser.add_argument('--max', action="store_true", help='synthesizing to max or mean num of training set. default is mean') 
    parser.add_argument('--no_mask', action="store_true", help='whether to mask the self class in sampling neighbor classes. default is mask')
    parser.add_argument('--gdc', type=str, choices=['ppr', 'hk', 'none'], default='ppr', help='how to convert to weighted graph')

    # def parse_baselines(baseline):
    #     baseline.parse_args(parser)
    #     for subcls in baseline.__subclasses__():
    #         parse_baselines(subcls)
    
    # parse_baselines(Baseline)

    args = parser.parse_args()

    return args, baseline
