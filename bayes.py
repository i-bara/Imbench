import os, json
from benchmark import all_config, args, benchmark

print(all_config)

if __name__ == '__main__':
    name = args.name
    config_file = 'benchmark/' + name + '.json'
    if os.path.isfile(config_file):
        with open(config_file) as f:
            config = json.load(f)
            for config_option in ['methods', 'datasets', 'seeds']:
                if len(config[config_option]) == 0:
                    config[config_option] = all_config[config_option]
            benchmark(name=name, methods=config['methods'], datasets=config['datasets'], seeds=config['seeds'])
    else:
        raise FileNotFoundError

