# -*- coding: utf-8 -*-



import torch
import dataset
import model


import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


H5_address ='/home/lyq/caffe-master/data_gaze/gaze_detection/h5/'
Save_model_address = '/disks/disk0/linyuqi/model/gaze_2eye_AR/'
BatchSize = 64 




gaze_model = model.gaze_model()
gaze_model = nn.DataParallel(gaze_model)
gaze_model.cuda()

img_list_left,img_list_right = dataset.load_all_h5(H5_address)
#print(img_list_left[1])
#print(img_list_right[1])
train_Dataset = dataset.train_gaze_dataset(img_list_left,img_list_right)
test_Dataset = dataset.test_gaze_dataset(img_list_left,img_list_right)
train_loader = torch.utils.data.DataLoader(train_Dataset,batch_size=BatchSize,num_workers=6)
test_loader = torch.utils.data.DataLoader(test_Dataset,batch_size=BatchSize,num_workers=6)


l1_loss = nn.SmoothL1Loss().cuda()
#l1_loss = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.01)

optimizer = torch.optim.SGD(gaze_model.parameters(),lr=0.001,momentum=0.9)


def accuracy_text(result, label):
    accuracy = 0
   
    for i in range(0,BatchSize):
        data_x = (-1) *(np.cos(result[i][0])) *(np.sin(result[i][1]))
        data_y = (-1) *(np.sin(result[i][0]))
        data_z = (-1) *(np.cos(result[i][0])) * (np.cos(result[i][1]))
        norm_data = np.sqrt(data_x * data_x + data_y * data_y + data_z * data_z)

        label_x = (-1) * (np.cos(label[i][0])) *(np.sin(label[i][1]))
        label_y = (-1) * (np.sin(label[i][0]))
        label_z = (-1) * (np.cos(label[i][0])) * (np.cos(label[i][1]))  
        norm_label = np.sqrt(label_x * label_x + label_y * label_y + label_z * label_z)
  
        angle_value = (data_x * label_x + data_y * label_y + data_z * label_z) / (norm_data * norm_label)
        accuracy += (np.arccos(angle_value) * 180) / 3.1415926

    accuracy_avg = accuracy / BatchSize
    return accuracy_avg

def train():
    gaze_model.train()
    for i ,(image_left,head_pose_left,label_left,image_right,head_pose_right,label_right) in enumerate(train_loader):
        image_left = image_left.squeeze(1)
        image_left = image_left.cuda()
        head_pose_left = head_pose_left.cuda()
        label_left = label_left.cuda()

        image_right = image_right.squeeze(1)
        image_right = image_right.cuda()
        head_pose_right = head_pose_right.cuda()
        label_right = label_right.cuda()

#        print(image_left[0][0])
#        print(head_pose.size())

        optimizer.zero_grad()
        result = gaze_model(image_left,head_pose_left,image_right,head_pose_right)

        label = (label_right+label_left)/2
        loss = l1_loss(result,label)
        
#        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i%20 ==0:
            print('num_of_batch = {}    train_loss={}'.format(i,loss))

def test():
    gaze_model.eval()
    for i ,(image_left,head_pose_left,label_left,image_right,head_pose_right,label_right) in enumerate(test_loader):
        image_left = image_left.squeeze(1)
        image_left = image_left.cuda()
        head_pose_left = head_pose_left.cuda()
        label_left = label_left.cuda()

        image_right = image_right.squeeze(1)
        image_right = image_right.cuda()
        head_pose_right = head_pose_right.cuda()
        label_right = label_right.cuda()
        
        
        
        with torch.no_grad():
            result = gaze_model(image_left,head_pose_left,image_right,head_pose_right)
        label = (label_right+label_left)/2
        loss = l1_loss(result,label)
        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
        
        if i%20 ==0:
            acc = accuracy_text(result,label)
            print('num_of_batch = {}    test_loss={}    accuracy={}   '.format(i,loss,acc))

        if i%100 ==0:
            for i in range(0,20):
                print('image_number:{:2}   process answer {:+5.4},{:+5.4})   label: {:+5.4},{:+5.4})'.format(i,result[i][0],result[i][1],label[i][0],label[i][1]))
            
def main():
    for epoch in range(1,30):
        print('epoch:' + str(epoch) + '\n')
        train()
        test()
        torch.save({'epoch':epoch,
                    'state_dict':gaze_model.state_dict(),},os.path.join(Save_model_address , 'checkpoint_{}.tar'.format(epoch)))
        
 
if __name__ =='__main__':
    main()       
