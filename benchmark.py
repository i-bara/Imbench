import os
import sys
import re
import json
import datetime
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
all_config['datasets'] = ['Cora_100', 'Cora_20', 'Cora_1', 
                          'CiteSeer_100', 'CiteSeer_20', 'CiteSeer_1', 
                          'PubMed_100', 'PubMed_20', 'PubMed_1', 
                          'chameleon_100', 'chameleon_20', 'chameleon_1', 
                          'squirrel_100', 'squirrel_20', 'squirrel_1', 
                          'Actor_100', 'Actor_20', 'Actor_1', 
                          'Wisconsin_100', 'Wisconsin_20', 'Wisconsin_1', 
                          'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS', 'ogbn-arxiv']
all_config['seeds'] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def config_args(method, dataset, seed):
    method_config = '--method ' + method
    if dataset.endswith('_100') or dataset.endswith('_20') or dataset.endswith('_1'):
        dataset_name, imb_ratio = dataset.split('_')
        dataset_config = f'--dataset {dataset_name} --imb_ratio {imb_ratio}'
    # if dataset == 'Cora_100':
    #     dataset_config = '--dataset Cora --imb_ratio 100'
    # elif dataset == 'Cora_20':
    #     dataset_config = '--dataset Cora --imb_ratio 20'
    # elif dataset == 'CiteSeer_100':
    #     dataset_config = '--dataset CiteSeer --imb_ratio 100'
    # elif dataset == 'CiteSeer_20':
    #     dataset_config = '--dataset CiteSeer --imb_ratio 20'
    # elif dataset == 'PubMed_100':
    #     dataset_config = '--dataset PubMed --imb_ratio 100'
    # elif dataset == 'PubMed_20':
    #     dataset_config = '--dataset PubMed --imb_ratio 20'
    # elif dataset == 'chameleon_100':
    #     dataset_config = '--dataset chameleon --imb_ratio 100'
    # elif dataset == 'chameleon_20':
    #     dataset_config = '--dataset chameleon --imb_ratio 20'
    # elif dataset == 'squirrel_100':
    #     dataset_config = '--dataset squirrel --imb_ratio 100'
    # elif dataset == 'squirrel_20':
    #     dataset_config = '--dataset squirrel --imb_ratio 20'
    # elif dataset == 'Actor_100':
    #     dataset_config = '--dataset Actor --imb_ratio 100'
    # elif dataset == 'Actor_20':
    #     dataset_config = '--dataset Actor --imb_ratio 20'
    # elif dataset == 'Wisconsin_100':
    #     dataset_config = '--dataset Wisconsin --imb_ratio 100'
    # elif dataset == 'Wisconsin_20':
    #     dataset_config = '--dataset Wisconsin --imb_ratio 20'
    elif dataset in ['Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS', 'ogbn-arxiv']:
        dataset_config = '--dataset ' + dataset + ' --imb_ratio 0'  # Do not adjust the imbalance
    else:
        raise NotImplementedError()
    return method_config + ' ' + dataset_config + ' --seed ' + seed.__str__()


def experiment(method, dataset, seed, options, records, records_file, cache_file):
    done = False
    for record in records:
        if record['method'] == method and record['dataset'] == dataset and record['seed'] == seed:
            done = True

    if not done:
        options_more = ''
        if not args.gpu:
            options_more += ' --device cpu'
        if args.debug:
            options_more += ' --debug'

        if args.debug:
            output = ' >> '
        else:
            output = ' > '

        command = "python src/main.py " + config_args(method=method, dataset=dataset, seed=seed) + " " + options + options_more + output + cache_file

        print('\n')
        print(command)

        begin_datetime = datetime.datetime.now()
        os.system(command)
        end_datetime = datetime.datetime.now()

        with open(cache_file) as f:
            info = f.read()
            match = re.match('[\\S\\s]*acc: ([\\d\\.]*), bacc: ([\\d\\.]*), f1: ([\\d\\.]*), auc: ([\\d\\.]*)\n', info)
            if match is not None:
                acc, bacc, f1, auc = tuple(map(lambda x: float(x), match.groups()))
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
                record['auc'] = auc
                print(record)
                records.append(record)

        if not args.debug:
            os.system("rm " + cache_file)
    
    if not args.debug:
        with open(records_file, 'w+') as f:
            json.dump(records, f, indent=4)


def benchmark(name, methods, datasets, seeds):
    suffix = ''

    if args.gpu:
        suffix += '_gpu'

    if args.debug:
        suffix += '_debug'
    
    if not os.path.isdir('log'):
        os.system('mkdir log')
    if not os.path.isdir('records'):
        os.system('mkdir records')
    if not os.path.isdir('cache'):
        os.system('mkdir cache')
    records_file = 'records/' + name + suffix + '.json'
    cache_file = 'cache/' + name + suffix + '.txt'

    if os.path.exists(records_file):
        with open(records_file) as f:
            records = json.load(f)
    else:
        records = []

    if os.path.exists(cache_file):
        os.system("rm " + cache_file)

    print(f'''
    benchmark_{name}

    methods = {methods}
    datasets = {datasets}
    seeds = {seeds}
    ''')

    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                experiment(method=method, dataset=dataset, seed=seed, options='', 
                           records=records, records_file=records_file, cache_file=cache_file)


def bayes(name, methods, datasets, seeds, options, iters):
    suffix = ''

    if args.gpu:
        suffix += '_gpu'

    if args.debug:
        suffix += '_debug'
    
    if not os.path.isdir('log'):
        os.system('mkdir log')
    if not os.path.isdir('records'):
        os.system('mkdir records')
    if not os.path.isdir('cache'):
        os.system('mkdir cache')
    cache_file = 'cache/' + name + suffix + '.txt'

    print(f'''
    benchmark_{name}

    methods = {methods}
    datasets = {datasets}
    seeds = {seeds}
    ''')

    for method in methods:
        for dataset in datasets:
            bayes_iter = 0
            while os.path.isfile(f'records/bayes_{name}{suffix}/{method}/{dataset}/{bayes_file + 1}_bayes.json'):
                bayes_iter += 1
            while bayes_iter < iters:
                bayes_file = f'records/bayes_{name}{suffix}/{method}/{dataset}/{bayes_file}_bayes.json'
                # TODO: Get options from file
                records_file = f'records/bayes_{name}{suffix}/{method}/{dataset}/{bayes_file}.json'

                if os.path.exists(records_file):
                    with open(records_file) as f:
                        records = json.load(f)
                else:
                    records = []
                for seed in seeds:
                    experiment(method=method, dataset=dataset, seed=seed, options=options, 
                            records=records, records_file=records_file, cache_file=cache_file)
                # TODO: Update options and state to file


if __name__ == '__main__':
    name = args.name
    config_file = 'benchmark/' + name + '.json'
    with open('dictionary.json') as f_dictionary:
        dictionary = json.load(f_dictionary)
        if os.path.isfile(config_file):
            with open(config_file) as f:
                config = json.load(f)  # A list of valid names
                for config_option in ['methods', 'datasets', 'seeds']:
                    if type(config[config_option]) != list:
                        if config[config_option] in dictionary[config_option]:  # Use dictionary.json
                            config[config_option] = dictionary[config_option][config[config_option]]
                        elif config[config_option] in all_config[config_option]:  # Any valid name
                            config[config_option] = [config[config_option]]
                        elif config[config_option] == 'all':  # 'all'
                            config[config_option] = all_config[config_option]
                        else:
                            raise NotImplementedError
                    elif len(config[config_option]) == 0:  # If it is an empty list, equivalent to 'all'
                        config[config_option] = all_config[config_option]
                benchmark(name=name, methods=config['methods'], datasets=config['datasets'], seeds=config['seeds'])
        else:
            raise FileNotFoundError
