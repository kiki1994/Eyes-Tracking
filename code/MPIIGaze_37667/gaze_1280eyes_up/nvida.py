# -*- coding:utf-8 -*-
#  !/usr/bin/env python
import os
import shutil
import sys
import pynvml

#mem_needed = int(sys.argv[1])
def get_useable_gpuID():
    pynvml.nvmlInit()
    for i in range(4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / 1024**2 >= 120:
            #print(i)
            return i
    return None
    
i = get_useable_gpuID
print(i)