# -*- coding:utf-8 -*-
#  !/usr/bin/env python
import os
import shutil
import sys
import pynvml

mem_needed = int(sys.argv[1])
def get_useable_gpuID():
    pynvml.nvmlInit()
    for i in range(2,4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / 1024**2 >= mem_needed:
            return i
    return None

def filter_tested(reference_listpath):
    with open(reference_listpath, 'r') as f:
        data = f.read().strip('\n').splitlines()
        data = list(filter(lambda x:x != '' and x!= '\n', data))
        data = list(map(lambda x:x.split(' ')[0], data))
        print('tested as below:\n', data)
    return data

def train_script():
    gpu_useableID = get_useable_gpuID()
    if gpu_useableID is None:
        print('no useable gpu!')
        return 
    dir_root = '/disks/disk0/linyuqi/dataset/data_gaze/eval_txts/'
    reset_txt(dir_root)
    txts = os.listdir(dir_root)
    txts.sort(key=lambda x:int(x[1:3]))
    for i, txt in enumerate(txts):
        data = filter_tested('./record.txt')
        person_id = txt.split('_')[0]
        if person_id not in data:
            print(person_id, gpu_useableID)
            os.system("python train_accu.py -g %d -p %s" % (gpu_useableID, person_id))

def reset_txt(path):
    for roots, dirs, files in os.walk(path):
        for ff in files:
            segs = ff.split("_")
            if segs[0] == 'test':
                source = os.path.join(roots, ff)
                dest = os.path.join(roots, "_".join(segs[1:]))
                shutil.move(source, dest)

if __name__ == '__main__':
    train_script()
# print(get_useable_gpuID())
