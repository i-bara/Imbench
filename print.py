import os
import re
import json
import datetime
import pandas as pd
from openpyxl import load_workbook
import statistics

method_list = ['vanilla', 'drgcn', 'smote', 'imgagn', 'ens', 'tam', 'lte4g', 'sann', 'sha', 'renode', 'pastel', 'hyperimba']
score_list = ['acc', 'bacc', 'f1', 'auc']
dataset_list = ['Cora_100', 'Cora_20', 'CiteSeer_100', 'CiteSeer_20', 'PubMed_100', 'PubMed_20', 'chameleon_100', 'chameleon_20', 'squirrel_100', 'squirrel_20', 'Actor_100', 'Actor_20', 'Wisconsin_100', 'Wisconsin_20', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS']
seed_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def get_method_offset(method):
    if method == 'vanilla':
        return 0
    elif method == 'drgcn':
        return 1
    elif method == 'smote':
        return 2
    elif method == 'imgagn':
        return 3
    elif method == 'ens':
        return 4
    elif method == 'tam':
        return 5
    elif method == 'lte4g':
        return 6
    elif method == 'sann':
        return 7
    elif method == 'sha':
        return 9
    elif method == 'renode':
        return 11
    elif method == 'pastel':
        return 12
    elif method == 'hyperimba':
        return 13
    else:
        raise NotImplementedError()


def get_score_offset(score):
    if score == 'acc':
        return 0
    elif score == 'bacc':
        return 1
    elif score == 'f1':
        return 2
    elif score == 'auc':
        return 3
    else:
        raise NotImplementedError()


def get_dataset_offset(dataset):
    if dataset == 'Cora_100':
        return (4, 2)
    elif dataset == 'Cora_20':
        return (4, 5)
    elif dataset == 'CiteSeer_100':
        return (4, 8)
    elif dataset == 'CiteSeer_20':
        return (4, 11)
    elif dataset == 'PubMed_100':
        return (4, 14)
    elif dataset == 'PubMed_20':
        return (4, 17)
    elif dataset == 'chameleon_100':
        return (4, 20)
    elif dataset == 'chameleon_20':
        return (4, 23)
    elif dataset == 'squirrel_100':
        return (4, 26)
    elif dataset == 'squirrel_20':
        return (4, 29)
    elif dataset == 'Actor_100':
        return (4, 32)
    elif dataset == 'Actor_20':
        return (4, 35)
    elif dataset == 'Wisconsin_100':
        return (4, 38)
    elif dataset == 'Wisconsin_20':
        return (4, 41)
    elif dataset == 'Amazon-Photo':
        return (22, 2)
    elif dataset == 'Amazon-Computers':
        return (22, 8)
    elif dataset == 'Coauthor-CS':
        return (22, 14)
    elif dataset == 'ogbn-arxiv':
        return (22, 20)
    else:
        raise NotImplementedError()
    
records_file_list = [
    # ('Photo', True),
    # ('Computers', True),
    # ('CS', True),
    ('0419vanilla', True),
    ('0419smote', True),
    ('0419ens', True),
    ('0419sha', True),
    # ('smote', True),
    # ('smote2', True),
    # ('ccp', False),
]

records = []

for records_file_item in records_file_list:
    if records_file_item[1]:
        records_file = 'records/' + records_file_item[0] + '_gpu.json'
        # records_file = 'records/records_gpu_' + records_file_item[0] + '.json'
    else:
        records_file = 'records/' + records_file_item[0] + '.json'
        # records_file = 'records/records_' + records_file_item[0] + '.json'
    if os.path.exists(records_file):
        with open(records_file) as f:
            records_this = json.load(f)
            records += records_this

benchmarks = dict()

for method in method_list:
    for dataset in dataset_list:
        benchmarks[(method, dataset)] = list()

for record in records:
    benchmarks[(record['method'], record['dataset'])].append((record['acc'], record['bacc'], record['f1'], record['auc']))

template_filename = 'Imbalance Benchmark Template.xlsx'
filename = 'Imbalance Benchmark.xlsx'

os.system('cp \'' + template_filename + '\' \'' + filename + '\'')
workbook = load_workbook(filename=filename)
ws4 = workbook["Node-level class imbalance "]

for method in method_list:
    for dataset in dataset_list:
        if len(benchmarks[(method, dataset)]) == len(seed_list):
            scores = dict()
            for score in score_list:
                scores[score] = list()
            for acc, bacc, f1, auc in benchmarks[(method, dataset)]:
                scores['acc'].append(acc)
                scores['bacc'].append(bacc)
                scores['f1'].append(f1)
                scores['auc'].append(auc)
            for score in score_list:
                mean_score = statistics.mean(scores[score])
                stdev_score = statistics.stdev(scores[score])
                row = 0
                column = 0
                row += get_method_offset(method=method)
                column += get_score_offset(score=score)
                row += get_dataset_offset(dataset=dataset)[0]
                column += get_dataset_offset(dataset=dataset)[1]
                ws4.cell(row=row, column=column).value = '%.3f (%.3f)' % (mean_score, stdev_score)
                # ws4.cell(row=row, column=column).value = mean_score.__str__() + ' (' + stdev_score.__str__() + ')'
            
workbook.save(filename=filename)
