import os
import re
import json
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


def experiment(records, records_file, cache_file, output_path, **kwargs):
    done = False
    for record in records:
        same = True
        for key, value in kwargs.items():
            if key not in record or record[key] != value:
                same = False
                break
        if same:
            done = True
            break
    
    if done:
        print('Have done!')
    else:
        options_more = ''
        if not args.gpu:
            options_more += ' --device cpu'
        if args.debug:
            options_more += ' --debug'

        if args.debug:
            output = ' >> '
        else:
            output = ' > '

        command = "python src/main.py"
        for key, value in kwargs.items():
            command += f' --{key} {value}'
        command += options_more + output + cache_file

        print()
        print(command)

        os.system(command)

        with open(cache_file) as f:
            info = f.read()
            match = re.match('[\\S\\s]*result: (\\{[\\S\\s]*\\})\n[\\S\\s]*', info)
            if match is not None:
                record = json.loads(match.groups()[0])
                print(record)
                records.append(record)

        if not args.debug:
            os.system("rm " + cache_file)
    
    if not args.debug:
        with open(records_file, 'w+') as f:
            json.dump(records, f, indent=4)


def benchmark(name, **config):
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
    if not os.path.isdir(f'records/{name}'):
        os.system(f'mkdir records/{name}')
    records_file = 'records/' + name + suffix + '.json'
    cache_file = 'cache/' + name + suffix + '.txt'
    output_path = f'records/{name}'

    if os.path.exists(records_file):
        with open(records_file) as f:
            records = json.load(f)
    else:
        records = []
        
    try:
        if os.path.exists(cache_file):
            os.system("rm " + cache_file)

        info = f'benchmark: {name}'
        print('=' * len(info))
        print(info)
        
        config_key_list = list(config.keys())
        current_config_index = [0] * len(config)
        
        while True:
            print('-' * len(info))
            current_config = {config_key_list[i]: config[config_key_list[i]][current_config_index[i]] for i in range(len(config))}
            for key, value in current_config.items():
                print(f'{key} = {value}')
            experiment(records=records, records_file=records_file, cache_file=cache_file, output_path=output_path,  **current_config)
            ptr = len(config) - 1
            while ptr >= 0:
                current_config_index[ptr] += 1
                if current_config_index[ptr] >= len(config[config_key_list[ptr]]):
                    current_config_index[ptr] = 0
                    ptr -= 1
                else:
                    break
            if ptr == -1:
                break
            
    except KeyboardInterrupt:
        with open(records_file, 'w+') as f:
            json.dump(records, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    name = args.name
    config_file = 'benchmark/' + name + '.json'
    with open('dictionary.json') as f_dictionary:
        dictionary = json.load(f_dictionary)
        if os.path.isfile(config_file):
            with open(config_file) as f:
                config = json.load(f)  # A list of valid names
                for key, value in config.items():
                    if not isinstance(value, list):
                        if key in dictionary and value in dictionary[key]:  # Use dictionary.json
                            config[key] = dictionary[key][value]
                        else:
                            config[key] = [value]
                    elif len(value) == 0:  # If it is an empty list, equivalent to 'all'
                        config[key] = dictionary[key]["all"]
                default = {
                    'method': 'vanilla',
                    'net': 'GCN',
                    
                    'dataset': 'Cora',
                    'split': 'lt',
                    'imb_ratio': 20,
                    
                    'seed': 100,
                }
                for key, value in default.items():
                    if key not in config:
                        config[key] = [value]
                benchmark(name=name, **config)
        else:
            raise FileNotFoundError
