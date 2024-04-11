import os
import datetime

t = datetime.datetime.now()
backup_dirs = ['records', 'benchmark']

if not os.path.isdir('backup'):
    os.system('mkdir backup')
if not os.path.isdir(f'backup/\'{t}\''):
    os.system(f'mkdir backup/\'{t}\'')

for backup_dir in backup_dirs:
    os.system(f'cp -r {backup_dir} backup/\'{t}\'/{backup_dir}/')
