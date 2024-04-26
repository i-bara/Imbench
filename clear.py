# import psutil

# for proc in psutil.process_iter():
#     try:
#         # this returns the list of opened files by the current process
#         flist = proc.open_files()
#         if flist:
#             print(proc.pid,proc.name)
#             for nt in flist:
#                 print("\t",nt.path)

#     # This catches a race condition where a process ends
#     # before we can examine its files    
#     except psutil.NoSuchProcess as err:
#         print("****",err)

import os
import datetime


files = os.listdir('log')
for file in files:
    file_path = os.path.join('log', file)
    if os.path.isfile(file_path):
        dt = datetime.datetime.strptime(file[:-14], r'%Y-%m-%d-%H-%M-%S')
        dt_now = datetime.datetime.now()
        if (dt_now - dt).days > 7:
            if not os.path.isdir('old_log'):
                os.system('mkdir old_log')
            old_file_path = os.path.join('old_log', file)
            os.system(f'mv {file_path} {old_file_path}')
