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
        
class BaseMode448(nn.Module):
    def __init__(self):
        super(BaseMode448,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=5,padding=0,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU())             #89
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=5,stride=3,padding=0,dilation=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU())             #28

        self.conv3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(128),
                         nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(128),
                         nn.ReLU(),
                         nn.MaxPool2d(2, stride=2))    #12

        self.conv5 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),
                         nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.BatchNorm2d(256),
                         nn.ReLU(),
                         nn.MaxPool2d(2, stride=2))    #4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        
    def forward(self,x):
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        x = x.view(x.size()[0], -1)
        return  x

class BaseMode1280(nn.Module):
    def __init__(self):
        super(BaseMode1280, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0, dilation=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=11, stride=4, padding=0, dilation=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())              #77*42

        self.conv3 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0, dilation=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())               #38*20
        self.conv4 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, dilation=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())                #38*20
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, dilation=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())                 #18*9
        self.conv6 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU())                  #18*9
        self.conv7 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU())                 #18*9
        self.conv8 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())                  #18*9
        self.conv9 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, dilation=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())                  # 8*4
        self.fc = nn.Sequential(nn.Linear(256*8*4,4096),
                                nn.ReLU(),
                                nn.Linear(4096,4096),
                                nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv9(self.conv8(self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))))
        x = x.view(x.size()[0], -1)
        x4096 = self.fc(x)
        return  x4096

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

class AR_Net(nn.Module):
    def __init__(self):
        super(AR_Net, self).__init__()
        self.AR_up = AR_Net_up()
        self.AR_down = AR_Net_down()
        self.fc4 = nn.Sequential(
            nn.Linear(1006,6))
    def forward(self, two_eyes,img_l,img_r ,head_pose_l,head_pose_r):
        imge_up = self.AR_up(two_eyes)
        imge_down = self.AR_down(img_l,img_r)
        result = self.fc4(torch.cat([imge_up,imge_down,head_pose_l,head_pose_r],1))
        return  result

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

    def forward(self, l,r):
        image_l = self.eyeModel_l(l)
        image_r = self.eyeModel_r(r)

        fc2_l = self.fc2_l(image_l)
        fc2_r = self.fc2_r(image_r)

        fc2_l_r = torch.cat([fc2_l, fc2_r], 1)
        fc3_l_r = self.fc3(fc2_l_r)
        return  fc3_l_r                              ##output 500

class AR_Net_up(nn.Module):
    def __init__(self):
        super(AR_Net_up, self).__init__()
        self.faceModel = BaseMode1280()                           # not sharing the weight

        self.fc1 = nn.Sequential(
            nn.Linear(4096,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())

    def forward(self, two_eyes):
        imag_2eyes = self.faceModel(two_eyes)

        fc_f = self.fc1(imag_2eyes)

        return  fc_f                       #output 1000
