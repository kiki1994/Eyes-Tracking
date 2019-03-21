# -*- coding: utf-8 -*-

import h5py
import numpy
import os
import os.path
import torch.utils.data as data
import torch

#H5_train_address = '/disks/disk0/linyuqi/dataset/data_gaze/eval_txts/'
#H5_test_address = '/disks/disk0/linyuqi/dataset/data_gaze/UT Multiview/test data/test_list/test_list.txt'

def judge(n):
    return n!='\n' and n!=''

def d_3(result):
    result = torch.from_numpy(result)
    data = torch.zeros([3])

    data[0] = (-1) * (torch.cos(result[0])) * (torch.sin(result[1]))

    data[1] = (-1) * (torch.sin(result[0]))

    data[2] = (-1) * (torch.cos(result[0])) * (torch.cos(result[1]))

    return data  # size 128*3

def load_all_h5(h5_adress):
    train_list = []
    for root, dirs, files in os.walk(h5_adress):
        for f in files:
            txt_absPath = os.path.join(root, f)
            ff = open(txt_absPath, 'r')
            path_list = ff.read().split('\n')
            path_list = filter(judge, path_list)            
            train_list.extend(path_list)
            ff.close()

    return train_list

def load_h5_list(txt_adress):
    ff = open(txt_adress, 'r')
    h5_list = ff.read().split('\n')
    h5_list = list(filter(judge, h5_list))
    ff.close()
    return h5_list
    
class gaze_train_dataset(data.Dataset):
    def __init__(self,h5_path_list):
        self.h5_path_list = h5_path_list[:]

    def __getitem__(self, index):
        filename = self.h5_path_list[index]
        f = h5py.File(filename, 'r')
        #face_img = f['face'].value
        left_eye_img = f['left_eye'].value
        right_eye_img = f['right_eye'].value
        left_label = f['left_label'].value
        right_label = f['right_label'].value
        left_headpose = f['left_head_poses'].value
        right_headpose = f['right_head_poses'].value
        f.close()
 
# Turn to 3D
        left_label = d_3(left_label)
        right_label = d_3(right_label)
        left_headpose = d_3(left_headpose)
        right_headpose = d_3(right_headpose)
        return  left_eye_img, right_eye_img, left_label, right_label, left_headpose, right_headpose   # , face

    def __len__(self):
        return len(self.h5_path_list)

class gaze_test_dataset(data.Dataset):
    def __init__(self,h5_path_list):
        self.h5_path_list = h5_path_list[:]

    def __getitem__(self, index):
        filename = self.h5_path_list[index]
        f = h5py.File(filename, 'r')
        #face_img = f['face'].value
        left_eye_img = f['data_l'].value
        right_eye_img = f['data_r'].value
        left_label = f['label_l'].value[0,:]
        right_label = f['label_r'].value[0,:]
        left_headpose = f['headpose_l'].value[0,:]
        right_headpose = f['headpose_r'].value[0,:]
        f.close()
 
# Turn to 3D

        return  left_eye_img, right_eye_img, left_label, right_label, left_headpose, right_headpose    # , face

    def __len__(self):
        return len(self.h5_path_list)
        
#if __name__ =='__main__':
#    list1 = load_all_h5(H5_train_address)
#    list2 = load_h5_list(H5_test_address)  
#    print(list1)
#    print(list2)