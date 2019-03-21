# -*- coding: utf-8 -*-

import h5py
import numpy
import os
import os.path
import torch.utils.data as data

def load_all_h5(root):
    left_list = []
    right_list = []
    for h5 in [root + a + '/' for a in os.listdir(root)]:
        if(h5.split('.')[-1] == 'h5'):
            person.append(h5)
    return person


class train_gaze_dataset(data.Dataset):
    def __init__(self,img_list):
        self.h5_path_list_left= img_list_left[:13]
       

    def __getitem__(self, index):
        filename_l = self.h5_path_list_left[index]
        f_l = h5py.File(filename_l, 'r')
        img_l = f_l['data'].value
        head_pose_l = f_l['label'].value[:, 2:4]
        label_l = f_l['label'].value[:, :2]     
        return head_pose_l,label_l,img_l

    def __len__(self):
        return len(self.h5_path_list_left)
        
class test_gaze_dataset(data.Dataset):
    def __init__(self,img_list_left,img_list_right):
        self.h5_path_list_left = img_list_left[13:]
        self.h5_path_list_right = img_list_right[13:]
    def __getitem__(self,index):
        filename_l = self.h5_path_list_left[index]
        f_l = h5py.File(filename_l, 'r')
        img_l = f_l['data'].value
        head_pose_l = f_l['labels'].value[0, 2:]
        label_l = f_l['labels'].value[0, :2]
        
        filename_r = self.h5_path_list_right[index]
        f_r = h5py.File(filename_r, 'r')
        img_r = f_r['data'].value
        head_pose_r = f_r['labels'].value[0, 2:]
        label_r = f_r['labels'].value[0, :2]
        return img_l, head_pose_l, label_l, img_r, head_pose_r, label_r

    def __len__(self):
        return len(self.h5_path_list_left)
        
