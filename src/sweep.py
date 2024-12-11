import wandb
import os
import json
from argparse import Namespace

from args import parse_args


def save():
    if not os.path.isdir('config'):
        os.mkdir('config')
        
    with open(config_filename, 'w') as f:
        json.dump(vars(config), f, indent=4)

try:
    config_name = 'Cora_20'

    config_filename = os.path.join('config', f'{config_name}.json')

    args, baseline = parse_args()

    try:
        with open(config_filename, 'r') as f:
            config = Namespace(**json.load(f))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        config = Namespace(
            project_name = config_name, 
            
            method = 'mixv2', 
            
            dataset = 'Cora', 
            data_path = 'datasets/', 
            imb_ratio = 20, 
            
            net = 'GCN', 
            
            debug = False, 
            device = 'cuda', 
            seed = 100, 
            
            n_layer = 2, 
            feat_dim = 64, 
            dropout = 0.5, 
            weight_decay = 0.0005, 
            epoch = 1000, 
            lr = 0.01, 
            warmup = 5, 
            keep_prob = 0.01, 
            tau = 2, 
            max = False, 
            no_mask = False, 
            proto_m = 0.99, 
            temp = 0.1, 
            lambda_pcon = 1, 
            topk = False, 
            k = 3, 
            epsilon = 0.05, 
            n_proto_per_cls = 4, 
            n_src_per_proto = 1, 
            n_mix_per_src = 10, 
            
            distance_influence_ratio = 1, 
            alpha = 100, 
        )

    sweep_config = {
        'method': 'random', 
        'metric': {
            'name': 'test_acc', 
            'goal': 'maximize', 
        }, 
        'parameters': {
            'n_layer': {
                'value': 2, 
            }, 
            'feat_dim': {
                'values': [128, 256, 512], 
            }, 
            'dropout': {
                'values': [0.5], 
            }, 
            'weight_decay': {
                'values': [5e-4], 
            }, 
            'epoch': {
                'value': 1000, 
            }, 
            'lr': {
                'values': [1e-2], 
            }, 
            'warmup': {
                'value': 10, 
            }, 
            'keep_prob': {
                'value': 0.01, 
            }, 
            'tau': {
                'values': [2], 
            }, 
            'max': {
                'value': False, 
            }, 
            'no_mask': {
                'value': True, 
            }, 
            'proto_m': {
                'values': [0.8, 0.95, 0.99], 
            }, 
            'temp': {
                'values': [0.01, 0.02], 
            }, 
            'lambda_pcon': {
                'distribution': 'uniform',
                'min': 0.73,
                'max': 0.74,
            }, 
            'topk': {
                'value': True, 
            }, 
            'k': {
                'values': [3, 4], 
            }, 
            'epsilon': {
                'values': [0.01, 0.02], 
            }, 
            'n_proto_per_cls': {
                'values': [4, 5], 
            }, 
            'n_src_per_proto': {
                'values': [2, 3], 
            }, 
            'n_mix_per_src': {
                'values': [8, 10], 
            }, 
            'distance_influence_ratio': {
                'distribution': 'uniform',
                'min': 1.8,
                'max': 2.5,
            }, 
            'alpha': {
                'values': [10, 50, 100], 
            }, 
        },
    }

    def train(config=config):
        print(type(wandb.config))
        print(config)
        baseline(config).run()

    sweep_id = wandb.sweep(sweep_config, project=config.project_name)
    wandb.agent(sweep_id, train, project=config.project_name, count=50)

    save()

except KeyboardInterrupt:
    save()
