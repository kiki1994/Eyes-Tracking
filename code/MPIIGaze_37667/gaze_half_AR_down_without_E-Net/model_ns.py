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
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=0,dilation=1),
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

class E_Net(nn.Module):
    def __init__(self):
        super(E_Net, self).__init__()
        self.probab_mode_l = BaseMode()
        self.probab_mode_r = BaseMode()
        self.fc5_l = nn.Sequential(
            nn.Linear(256*1*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())
        self.fc5_r = nn.Sequential(
            nn.Linear(256*1*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())
        self.fc6 = nn.Sequential(
            nn.Linear(1000,2),
            nn.Softmax(dim=1))
    def forward(self, img_l,img_r):
        imge_l = self.probab_mode_l(img_l)
        imge_r = self.probab_mode_r(img_r)

        img_pro_l = self.fc5_l(imge_l)
        img_pro_r = self.fc5_r(imge_r)

        pro_l_r = self.fc6(torch.cat([img_pro_l,img_pro_r],1))
#        pro_l_r = nn.Softmax(pro,dim=1)
        return pro_l_r                   ##output 2 probability



class AR_Net_down(nn.Module):
    def __init__(self):
        super(AR_Net_down, self).__init__()
        self.eyeModel_l = BaseMode()
        self.eyeModel_r = BaseMode()

        self.fc2_l = nn.Sequential(
            nn.Linear(256 * 1 * 4, 500),
            nn.ReLU())
        self.fc2_r = nn.Sequential(
            nn.Linear(256 * 1 * 4, 500),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(506,6))

    def forward(self, l,r,h_l,h_r):
        image_l = self.eyeModel_l(l)
        image_r = self.eyeModel_r(r)

        fc2_l = self.fc2_l(image_l)
        fc2_r = self.fc2_r(image_r)

        fc2_l_r = torch.cat([fc2_l, fc2_r], 1)
        fc3_lr = self.fc3(fc2_l_r)
        result = self.fc(torch.cat([fc3_lr,h_l,h_r],1))
        return  result                           ##output 500

