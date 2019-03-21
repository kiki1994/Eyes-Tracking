# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data 
import math
import torch.nn.functional as F

class BaseMode(nn.Module):
    def __init__(self):
        super(BaseMode,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.MaxPool2d(2,stride=2))

        self.conv3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(128),
                         nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(128),
                         nn.ReLU(),
                         nn.MaxPool2d(2, stride=2))

        self.conv5 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),
                         nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),
                         nn.ReLU(),
                         nn.MaxPool2d(2, stride=2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        
    def forward(self,x):
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        x = x.view(x.size()[0], -1)
        return  x
modle = BaseMode()
#parem = list(modle.parameters())
for name, param in modle.named_parameters():
    print(name, ':', param)
