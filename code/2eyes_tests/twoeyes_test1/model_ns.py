# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data 
import math
import numpy as np
import torch.nn.functional as F

class BaseMode(nn.Module):
    def __init__(self):
        super(BaseMode,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,96,kernel_size=[1,2],stride=[1,2],padding=0,dilation=1),
                         nn.BatchNorm2d(96),
                         nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(96,128,kernel_size=[1,2],stride=[1,2],padding=0,dilation=1),
                         nn.BatchNorm2d(128),       #60*60
                         nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,dilation=1),
                         nn.BatchNorm2d(256),        #60*60
                         nn.ReLU(),
                         nn.MaxPool2d(3, stride=2))
        self.conv5 = nn.Sequential(nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1,dilation=1),
                         nn.BatchNorm2d(384),             #29*29
                         nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1,dilation=1),
                         nn.BatchNorm2d(384),         #29
                         nn.ReLU(),
                         nn.MaxPool2d(3, stride=2))  #14
        self.conv7 = nn.Sequential(nn.Conv2d(384,256, kernel_size=3,stride=1, padding=1, dilation =1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),
                          nn.MaxPool2d(3, stride=2))  #6
        self.conv8 = nn.Sequential(nn.Conv2d(256,256, kernel_size=3,stride=1, padding=0, dilation =1),
                          nn.BatchNorm2d(256),        #4
                          nn.ReLU())
        self.fc1 = nn.Sequential(
                   nn.Linear(256*4*4,4096),
                  # nn.Dropout(0.5),
                   nn.ReLU())
        self.fc2 = nn.Sequential(
                   nn.Linear(4096,1000),
                  # nn.Dropout(0.5),
                   nn.ReLU())
        self.fc3 = nn.Sequential(
                   nn.Linear(1084,500))
        self.fc4 = nn.Sequential(
                   nn.Linear(500,6))
                   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        
    def forward(self, x, eyes_loc, face_loc, left_headpose, right_headpose):
        x = self.conv8(self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv1(x)))))))
        x = x.view(x.size()[0], -1)
        xf = self.fc2(self.fc1(x))
        result = self.fc4(self.fc3(torch.cat([xf, eyes_loc, face_loc, left_headpose, right_headpose],1)))
        return  result

def Spatialweights(facemap, fcface):
    weightmap = torch.zeros([128,256,12,12]).cuda()
    weightmap.requries_grad = True
#    print(facemap.shape[1])
    for i in range(facemap.shape[1]):
        weightmap[0,i,:,:] = torch.mul(facemap[0, i, :, :],fcface[0,0,:,:])
       # print(facemap[0,i,:,:])            #(1, 256, 13, 13)
#    print(weightmap.shape)
    return weightmap
    
    
