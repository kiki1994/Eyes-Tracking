# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data 
import math

class gaze_model(nn.Module):
    def __init__(self):
        super(gaze_model,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,20,kernel_size=5,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(20),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(2,stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(20,50,kernel_size=5,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(50),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(2,stride=2))
        ''''
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64,50,kernel_size=3,stride=2,padding=1,dilation=1),
                         nn.BatchNorm2d(50),
                         nn.ReLU(inplace=True))
         '''
        self.fc1 = nn.Linear(3600,500)
        self.fc2 = nn.Linear(502,2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self,img,head_pose):
        batch_size = img.size()[0]
#        img9_15 = self.conv4(self.conv3(self.conv2(self.conv1(img))))
        img6_12 = self.conv2((self.conv1(img)))
        img6_12 = img6_12.view(batch_size,-1)
#        print(img6_12.size())
        fc1 = self.fc1(img6_12)
        result = self.fc2(torch.cat([fc1,head_pose],1))
        return result
    
        
