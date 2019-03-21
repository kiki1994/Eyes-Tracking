# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data 
import math

class gaze_model(nn.Module):
    def __init__(self):
        super(gaze_model,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(2,stride=2))

        self.conv3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(128),
                         nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(128),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(2, stride=2))

        self.conv5 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),
                         nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Linear(1024,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(1504, 2)
       
        self.fc11 = nn.Linear(1024,500)
#        self.fc22 = nn Linear(1000,1000)
        self.fc33 = nn.Linear(1000,500)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self,img_l,head_pose_l,img_r,head_pose_r):
        batch_size_l = img_l.size()[0]
        batch_size_r = img_r.size()[0]
        img1_4_l = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(img_l))))))
        img1_4_r = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(img_r))))))
#        img1_4_l = self.conv2(self.conv1(img_l))

        img1_4_l = img1_4_l.view(batch_size_l,-1)
        img1_4_r = img1_4_r.view(batch_size_r,-1)

#        print(img1_4_l.size())
        fc2_l = self.fc2(self.fc1(img1_4_l))
        fc2_r = self.fc2(self.fc1(img1_4_r))

        fc11_l = self.fc11(img1_4_l)
        fc11_r = self.fc11(img1_4_l)

        fc22_result = self.fc33(torch.cat([fc11_l,fc11_r],1))        
        result = self.fc3(torch.cat([fc2_r,fc2_l,fc22_result, head_pose_l,head_pose_r],1))

        return result
            
