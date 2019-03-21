# -*- coding: utf-8 -*-

import h5py
import numpy
import os
import os.path
import torch.utils.data as data

def load_all_h5(root):
    list = []
    for person in [root + a + '/' for a in os.listdir(root)]:
        for day in [person + a + '/' + 'left/'for a in os.listdir(person)]:
            for file in os.listdir(day):
                if(file.split('.')[-1] == 'h5'):
                    list.append(day + file)
    return list

class train_gaze_dataset(data.Dataset):
    def __init__(self,img_list):
        self.h5_path_list = img_list[:200000]
        
    def __getitem__(self,index):
        filename = self.h5_path_list[index]
        f = h5py.File(filename,'r')
        img = f['data'].value
        head_pose = f['labels'].value[0,2:]
        label = f['labels'].value[0,:2]
        return img, head_pose, label
    
    def __len__(self):
        return len(self.h5_path_list)
        
class test_gaze_dataset(data.Dataset):
    def __init__(self,img_list):
        self.h5_path_list = img_list[200000:]
        
    def __getitem__(self,index):
        filename = self.h5_path_list[index]
        f = h5py.File(filename,'r')
        img = f['data'].value
        head_pose = f['labels'].value[0,2:]
        label = f['labels'].value[0,:2]
        return img, head_pose, label
    
    def __len__(self):
        return len(self.h5_path_list)
        
