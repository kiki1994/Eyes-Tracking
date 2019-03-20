# -*- coding: utf-8 -*-

import h5py
import numpy
import os
import os.path
import torch.utils.data as data
import torch
def judge(n):
    return n!='\n' and n!=''

def d_3(result):
    result = torch.from_numpy(result)
    data = torch.zeros([3])

    data[0] = (-1) * (torch.cos(result[0])) * (torch.sin(result[1]))

    data[1] = (-1) * (torch.sin(result[0]))

    data[2] = (-1) * (torch.cos(result[0])) * (torch.cos(result[1]))

    return data  # size 128*3

#def load_all_h5(h5_adress):
#    test_list = []
#    train_list = []
#   for root, dirs, files in os.walk(h5_adress):
#        for f in files:
#           txt_absPath = os.path.join(root, f)
#           path_list = ff.read().split('\n')
#            path_list = filter(judge, path_list)
#            segs = f.split("_")
#            if segs[0] != "test":
#                train_list.extend(path_list)
#            else:
#                test_list.extend(path_list)
#            ff.close()

#    return train_list, test_list

def load_h5_list(txt_adress):
    ff = open(txt_adress, 'r')
    h5_list = ff.read().split('\n')
    h5_list = list(filter(judge, h5_list))
    ff.close()
    return h5_list
    
    
class gaze_dataset(data.Dataset):
    def __init__(self,h5_path_list):
        self.h5_path_list = h5_path_list[:]

    def __getitem__(self, index):
        filename = self.h5_path_list[index]
        f = h5py.File(filename, 'r')
        face_img = f['face'].value
#        left_eye_img = f['left_eye'].value
#        right_eye_img = f['right_eye'].value
        left_label = f['left_gaze'].value[:]
        right_label = f['right_gaze'].value[:]
        left_headpose = f['left_headpose'].value[:]
        right_headpose = f['right_headpose'].value[:]
        f.close()
 
# Turn to 3D
        left_label = d_3(left_label)
        right_label = d_3(right_label)
        left_headpose = d_3(left_headpose)
        right_headpose = d_3(right_headpose)
        return face_img, left_label, right_label, left_headpose, right_headpose 

    def __len__(self):
        return len(self.h5_path_list)
        
