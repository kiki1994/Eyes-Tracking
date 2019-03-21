# -*- coding: utf-8 -*-

import h5py
import numpy
import os
import os.path
import torch.utils.data as data

def load_all_h5(root):
    name_list = []
    for person in [root + a + '/' for a in os.listdir(root)]:
        for day in [person + a + '/' for a in os.listdir(person)]:
            for file_name in os.listdir(day):
                if(file_name.split('.')[-1] == 'h5'):
                   name_list.append(day + file_name)

    return name_list


class train_gaze_dataset(data.Dataset):
    def __init__(self,img_name_list):
        self.h5_path_list = img_name_list[:200000]

    def __getitem__(self, index):
        filename = self.h5_path_list[index]
        f_l = h5py.File(filename, 'r')
        face_img = f_l['data2'].value
        l_eye_img = f_l['datal'].value
        r_eye_img = f_l['datar'].value
        label_l = f_l['labels'].value[0, :2]
        head_pose_l = f_l['labels'].value[0, 2:4]
        label_r = f_l['labels'].value[0, 4:6]
        head_pose_r = f_l['labels'].value[0, 6:8]

        return face_img, l_eye_img, r_eye_img, label_l, label_r, head_pose_l, head_pose_r

    def __len__(self):
        return len(self.h5_path_list)
        
class test_gaze_dataset(data.Dataset):
    def __init__(self,img_name_list):
        self.h5_path_list = img_name_list[200000:]

    def __getitem__(self,index):
        filename = self.h5_path_list[index]
        f_l = h5py.File(filename, 'r')
        face_img = f_l['data2'].value
        l_eye_img = f_l['datal'].value
        r_eye_img = f_l['datar'].value
        label_l = f_l['labels'].value[0, :2]
        head_pose_l = f_l['labels'].value[0, 2:4]
        label_r = f_l['labels'].value[0, 4:6]
        head_pose_r = f_l['labels'].value[0, 6:8]
        

        return face_img, l_eye_img, r_eye_img, label_l, label_r, head_pose_l, head_pose_r

    def __len__(self):
        return len(self.h5_path_list)
        
