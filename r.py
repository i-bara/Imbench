"""
Run in slurm using gpu
Usage: python run.py <config_name>
"""

import os
import argparse
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()

    return args


args = parse_args()

if not os.path.isdir('log'):
    os.system('mkdir log')

print(f'[{datetime.datetime.now()}]: Begin!')

for name in args.names:
    command = f'python benchmark.py {name} --gpu > log/"$(date +%Y-%m-%d-%H-%M-%S-%N)".log 2>&1'
    print(f'[{datetime.datetime.now()}]: {command}    ', end='')
    os.system(command)
    print(f'OK')

print(f'[{datetime.datetime.now()}]: Finish!')
os.system('python vgg.py')
print('waiting quit...')
