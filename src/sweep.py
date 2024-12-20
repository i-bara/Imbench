"""
Sweep config for wandb sweeps

Usage:
    python sweep.py --method <method>
    
    <method>: default mix
"""

import os
import json
from argparse import Namespace

import wandb
import torch

from args import parse_args


# def save():
#     if not os.path.isdir('config'):
#         os.mkdir('config')
        
#     with open(config_filename, 'w') as f:
#         json.dump(vars(config), f, indent=4)

try:
    config_name = 'Cora_20'

    config_filename = os.path.join('config', f'{config_name}.json')

    args, baseline = parse_args()

    # try:
    #     with open(config_filename, 'r') as f:
    #         config = Namespace(**json.load(f))
    # except (FileNotFoundError, json.decoder.JSONDecodeError):
    #     config = Namespace(
    #         project_name = config_name, 
            
    #         # Method
    #         method = 'mix', 
            
    #         net = 'GCN', 
            
    #         # Dataset
    #         dataset = 'Cora', 
    #         split = 'lt', 
    #         imb_ratio = 100, 
            
    #         data_path = 'datasets/', 
            
    #         debug = False, 
    #         device = 'cuda', 
    #         seed = 100, 
            
    #         # Model parameters
    #         n_layer = 2, 
    #         feat_dim = 64, 
            
    #         # Training parameters
    #         dropout = 0.5, 
    #         weight_decay = 0.0005, 
    #         epoch = 1000, 
    #         lr = 0.01, 
            
    #         warmup = 5, 
    #         keep_prob = 0.01, 
    #         tau = 2, 
    #         max = False, 
    #         no_mask = False, 
    #         proto_m = 0.99, 
    #         temp = 0.1, 
    #         lambda_pcon = 1, 
    #         topk = False, 
    #         k = 3, 
    #         epsilon = 0.05, 
    #         n_proto_per_cls = 4, 
    #         n_src_per_proto = 1, 
    #         n_mix_per_src = 10, 
            
    #         distance_influence_ratio = 1, 
    #         distance_p = 1, 
    #         distance_multiplier = 1, 
    #         distance_offset = 0, 
    #         alpha = 100, 
    #     )

    sweep_config = {
        'method': 'bayes', 
        'metric': {
            'name': 'test_acc', 
            'goal': 'maximize', 
        }, 
        'parameters': {
            'n_layer': {
                'values': [2], 
                # 'values': [1, 2, 3], 
            }, 
            'feat_dim': {
                'values': [256], 
                # 'values': [64, 128, 256, 512, 1024], 
            }, 
            'dropout': {
                'values': [0.5], 
                # 'distribution': 'uniform',
                # 'min': 0.3,
                # 'max': 0.85,
            }, 
            'weight_decay': {
                'values': [5e-4], 
                # 'distribution': 'log_uniform_values',
                # 'min': 5e-4,
                # 'max': 5e-2,
            }, 
            'epoch': {
                'value': 1000, 
            }, 
            'lr': {
                'values': [1e-3], 
                # 'distribution': 'log_uniform_values',
                # 'min': 1e-3,
                # 'max': 1e-1,
            }, 
            'warmup': {
                'values': [20], 
                # 'distribution': 'int_uniform',
                # 'min': 1,
                # 'max': 50,
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
                'values': [0.99], 
            }, 
            'temp': {
                # 'values': [0.02], 
                'distribution': 'log_uniform_values',
                'min': 0.002,
                'max': 2,
            }, 
            'lambda_pcon': {
                'values': [0.74], 
                # 'distribution': 'uniform',
                # 'min': 0.3,
                # 'max': 3,
            }, 
            'topk': {
                'value': True, 
            }, 
            'k': {
                'values': [3], 
            }, 
            'epsilon': {
                'values': [0.02], 
            }, 
            'n_proto_per_cls': {
                'values': [6, ], 
                # 'distribution': 'int_uniform',
                # 'min': 4,
                # 'max': 12,
                # 'values': [4, 5, 6], # 4
            }, 
            'n_src_per_proto': {
                'values': [2], 
                # 'distribution': 'int_uniform',
                # 'min': 1,
                # 'max': 4,
            }, 
            'n_mix_per_src': {
                'values': [12], 
                # 'distribution': 'int_uniform',
                # 'min': 9,
                # 'max': 15,
                # 'values': [9, 10, 11, 12, 13, 14, 15], 
                # 'values': [6, 8, 10, 12], 
            }, 
            'distance_influence_ratio': {
                'values': [1.5],
                # 'distribution': 'uniform',
                # 'min': 0.6,
                # 'max': 2.4,
            }, 
            'distance_p': {
                'values': [1.4],
                # 'distribution': 'uniform',
                # 'min': 0.7,
                # 'max': 2.6,
            }, 
            'distance_multiplier': {
                'values': [2],
                # 'distribution': 'log_uniform_values',
                # 'min': 0.75,
                # 'max': 4,
            }, 
            'distance_offset': {
                'values': [5],
                # 'distribution': 'uniform',
                # 'min': -10,
                # 'max': 10,
            }, 
            'alpha': {
                'values': [200], # Cora 60
                # 'distribution': 'log_uniform_values',
                # 'min': 1,
                # 'max': 1000,
            }, 
        },
    }

    def train(config=args):
        print(type(wandb.config))
        print(config)
        print(baseline)
        torch.cuda.empty_cache()
        baseline(config).run()

    # sweep_id = wandb.sweep(sweep_config, project=config_name)
    wandb.agent('96men15m', train, project=config_name, count=50)

    # save()

except KeyboardInterrupt:
    # save()
    pass
