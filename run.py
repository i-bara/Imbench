"""
Run in slurm using gpu
Usage: python run.py <config_name>
"""

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('name')
    args = parser.parse_args()

    return args


args = parse_args()

if os.path.isdir('log'):
    os.system('mkdir log')

command = f'srun -p dell --gres=gpu:V100:1 --time=23:59:59 python benchmark.py {args.name} --gpu > log/"$(date +%Y-%m-%d-%H-%M-%S-%N)".log 2>&1'
print(f'Run: {command}')
os.system(command)
print(f'End')
