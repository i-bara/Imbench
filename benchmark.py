import os
import sys
import re
import json
import datetime
import pandas as pd
from openpyxl import load_workbook
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('name')
    parser.add_argument('--gpu', action='store_true',
                        help='use gpu')
    parser.add_argument('--debug', action='store_true',
                        help='only for debugging, do not save as records')
    args = parser.parse_args()

    return args


args = parse_args()

all_config = dict()
all_config['methods'] = ['vanilla', 'drgcn', 'smote', 'imgagn', 'ens', 'tam', 'lte4g', 'sann', 'sha', 'renode', 'pastel', 'hyperimba']
all_config['datasets'] = ['Cora_100', 'Cora_20', 'CiteSeer_100', 'CiteSeer_20', 'PubMed_100', 'PubMed_20', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS']
all_config['seeds'] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def config_args(method, dataset, seed):
    method_config = '--method ' + method
    if dataset == 'Cora_100':
        dataset_config = '--dataset Cora --imb_ratio 100'
    elif dataset == 'Cora_20':
        dataset_config = '--dataset Cora --imb_ratio 20'
    elif dataset == 'CiteSeer_100':
        dataset_config = '--dataset CiteSeer --imb_ratio 100'
    elif dataset == 'CiteSeer_20':
        dataset_config = '--dataset CiteSeer --imb_ratio 20'
    elif dataset == 'PubMed_100':
        dataset_config = '--dataset PubMed --imb_ratio 100'
    elif dataset == 'PubMed_20':
        dataset_config = '--dataset PubMed --imb_ratio 20'
    elif dataset in ['Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS']:
        dataset_config = '--dataset ' + dataset + ' --imb_ratio 0'  # Do not adjust the imbalance
    else:
        raise NotImplementedError()
    return method_config + ' ' + dataset_config + ' --seed ' + seed.__str__()


def experiment(method, dataset, seed, records, records_file, cache_file):
    done = False
    for record in records:
        if record['method'] == method and record['dataset'] == dataset and record['seed'] == seed:
            done = True

    if not done:
        if args.gpu:
            command = "python src/main.py " + config_args(method=method, dataset=dataset, seed=seed) + " > " + cache_file
        else:
            command = "python src/main.py " + config_args(method=method, dataset=dataset, seed=seed) + " --device cpu > " + cache_file
        print('\n')
        print(command)

        begin_datetime = datetime.datetime.now()
        os.system(command)
        end_datetime = datetime.datetime.now()

        with open(cache_file) as f:
            info = f.readline()
            match = re.match('acc: ([\\d\\.]*), bacc: ([\\d\\.]*), f1: ([\\d\\.]*)', info)
            if match is not None:
                acc, bacc, f1 = tuple(map(lambda x: float(x), match.groups()))
                record = dict()
                record['begin_datetime'] = begin_datetime.__str__()
                record['end_datetime'] = end_datetime.__str__()
                record['time_erased'] = (end_datetime - begin_datetime).__str__()
                record['method'] = method
                record['dataset'] = dataset
                record['seed'] = seed
                record['acc'] = acc
                record['bacc'] = bacc
                record['f1'] = f1
                print(record)
                records.append(record)

        if not args.debug:
            os.system("rm " + cache_file)
    
    if not args.debug:
        with open(records_file, 'w+') as f:
            json.dump(records, f, indent=4)


def benchmark(name, methods, datasets, seeds):
    suffix = '_'

    if args.gpu:
        suffix += 'gpu_'

    if args.debug:
        suffix += 'debug_'
    
    if not os.path.isdir('log'):
        os.system('mkdir log')
    if not os.path.isdir('records'):
        os.system('mkdir records')
    if not os.path.isdir('cache'):
        os.system('mkdir cache')
    records_file = 'records/records' + suffix + name + '.json'
    cache_file = 'cache/cache' + suffix + name + '.txt'

    if os.path.exists(records_file):
        with open(records_file) as f:
            records = json.load(f)
    else:
        records = []

    print(f'''
    benchmark_{name}

    methods = {methods}
    datasets = {datasets}
    seeds = {seeds}
    ''')

    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                experiment(method=method, dataset=dataset, seed=seed, 
                           records=records, records_file=records_file, cache_file=cache_file)


if __name__ == '__main__':
    name = sys.argv[1]
    config_file = 'benchmark/' + name + '.json'
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            for config_option in ['methods', 'datasets', 'seeds']:
                if len(config[config_option]) == 0:
                    config[config_option] = all_config[config_option]
            benchmark(name=name, methods=config['methods'], datasets=config['datasets'], seeds=config['seeds'])
    else:
        raise FileNotFoundError
