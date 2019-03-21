import numpy
import os
import os.path
import torch.utils.data as data


h5_adress = '/disks/disk0/linyuqi/dataset/data_gaze/gaze_detection/h5_2eyes/'
test_list = []
train_list = []
for root, dirs, files in os.walk(h5_adress):
    print(root,dirs,files)
    for f in files:
        txt_absPath = os.path.join(root, f)
        #ff = open(txt_absPath, 'r')
        #path_list = ff.read().split('\n')
        if txt_absPath.split("_")[0] != "train":
            test_list.extend(path_list)
        else:
            train_list.extend(path_list)
        f.close()


