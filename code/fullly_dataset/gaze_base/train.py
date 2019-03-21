# -*- coding: utf-8 -*-



import torch
import dataset
import model

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


H5_address ='/home/lyq/caffe-master/data_gaze/gaze_detection/h5/'
Save_model_address = '/disks/disk0/linyuqi/model/gaze_demo/'





gaze_model = model.gaze_model()
#gaze_model = nn.DataParallel(gaze_model)
gaze_model.cuda()

img_list = dataset.load_all_h5(H5_address)
train_Dataset = dataset.train_gaze_dataset(img_list)
test_Dataset = dataset.test_gaze_dataset(img_list)
train_loader = torch.utils.data.DataLoader(train_Dataset,batch_size=128,num_workers=6)
test_loader = torch.utils.data.DataLoader(test_Dataset,batch_size=128,num_workers=6)

l1_loss = nn.SmoothL1Loss().cuda()
optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.001)


def train():
    gaze_model.train()
    for i ,(image,head_pose,label) in enumerate(train_loader):
        image = image.squeeze(1)
        image = image.cuda()
        head_pose = head_pose.cuda()
        label = label.cuda()
#        print(image.size())
#        print(head_pose.size())
        result = gaze_model(image,head_pose)
        
        loss = l1_loss(result,label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%20 ==0:
            print('num_of_batch = {}    train_loss={}'.format(i,loss))

def test():
    gaze_model.eval()
    for i ,(image,head_pose,label) in enumerate(test_loader):
        image = image.squeeze(1)
        image = image.cuda()
        head_pose = head_pose.cuda()
        label = label.cuda()
        
        
        
        with torch.no_grad():
            result = gaze_model(image,head_pose)
        loss = l1_loss(result,label)
        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
        
        if i%20 ==0:
            print('num_of_batch = {}    test_loss={}'.format(i,loss))
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
