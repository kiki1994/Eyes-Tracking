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
        self.conv1 = nn.Sequential(nn.Conv2d(1,96,kernel_size=11,stride=4,padding=0,dilation=1),
                         nn.BatchNorm2d(96),        #110
                         nn.ReLU(),
                         nn.MaxPool2d(3,stride=2))   #54
        self.conv2 = nn.Sequential(nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2,dilation=1),
                         nn.BatchNorm2d(256),       #54
                         nn.ReLU(),
                         nn.MaxPool2d(3,stride=2))   #26
        self.conv3 = nn.Sequential(nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1,dilation=1),
                         nn.BatchNorm2d(384),       #26
                         nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1,dilation=1),
                         nn.BatchNorm2d(384),        #26
                         nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1,dilation=1),
                         nn.BatchNorm2d(256),             #26
                         nn.ReLU(),
                          nn.MaxPool2d(3, stride=2))      #12
##################### DSPP #####################       
        self.fc6_dspp = nn.Sequential(nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),         #12
                         nn.ReLU())
        self.fc7_dspp = nn.Sequential(nn.Conv2d(256,256, kernel_size=1,stride=1, padding=0, dilation =1),
                          nn.BatchNorm2d(256),        #12
                          nn.ReLU())
        self.fc8_dspp = nn.Sequential(nn.Conv2d(256,1, kernel_size=1,stride=1, padding=0, dilation =1),
                          nn.BatchNorm2d(1),        #12
                          nn.ReLU())
        self.fc6 = nn.Sequential(
                   nn.Linear(256*12*12,4096),
                   nn.Dropout(0.5),
                   nn.ReLU())
        self.fc7 = nn.Sequential(
                   nn.Linear(4096,4096),
                   nn.Dropout(0.5),
                   nn.ReLU())
        self.fc8 = nn.Sequential(
                   nn.Linear(4096,3))
                   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        
    def forward(self,x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x_dspp = self.fc8_dspp(self.fc7_dspp(self.fc6_dspp(x)))
        v = x*x_dspp
        v = v.view(v.size()[0], -1)
        
        result = self.fc8(self.fc7(self.fc6(v)))
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
    
    
