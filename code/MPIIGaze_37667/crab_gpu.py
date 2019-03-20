
import os
import sys
import pynvml
import time

def print_ts(message):
    print ("[%s] %s"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
def run(interval, command):
    print_ts("-"*100)
    print_ts("Command %s"%command)
    print_ts("Starting every %s seconds."%interval)
    print_ts("-"*100)
    while True:
        try:
            # sleep for the remaining seconds of interval
            time_remaining = interval-time.time()%interval
            print_ts("Sleeping until %s (%s seconds)..."%((time.ctime(time.time()+time_remaining)), time_remaining))
            time.sleep(time_remaining)
            print_ts("Starting command.")
            # execute the command
            #i = get_useable_gpuID()
            status = train_script(command)
            if status == 0:
                break
            print_ts("-"*100)
            print_ts("Command status = %s."%status)
            print("allalalaalalalallalalal")
        except (Exception, e):
            print (e)
            
def get_useable_gpuID():
    pynvml.nvmlInit()
    for i in range(4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / 1024**2 >= 5000:
            return i
    return None
   
def train_script(c):
    gpu_useableID = get_useable_gpuID()
    if gpu_useableID is None:
        print('no useable gpu!')
        return 
    
    else:
        status = os.system(c)
        return status
              

if __name__=="__main__":
    interval = 5
    command = r"ls"
    run(interval, command)
