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
        super(BaseMode, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, dilation=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, stride=2))  #(64,14,59)
        self.res2a_banch1 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm2d(256))  # 14,59
        self.res2a_banch2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm2d(64),  # 14ï¼?9
                                   nn.ReLU())
        self.res2a_banch2b = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(64),  # 14ï¼?9
                                   nn.ReLU())
        self.res2a_banch2c = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm2d(256))  # 14,59


        self.res2b_banch2a = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm2d(64),  # 14,59
                                   nn.ReLU())
        self.res2b_banch2b = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.res2b_banch2c = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm2d(256))
        self.res2c_banch2a = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(64),  # 14,59
                                           nn.ReLU())
        self.res2c_banch2b = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU())
        self.res2c_banch2c = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(256))


        self.res3a_banch1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, dilation=1),
                                          nn.BatchNorm2d(512))  # 7ï¼?0
        self.res3a_banch2a = nn.Sequential(nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=2, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3a_banch2b = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3a_banch2c = nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(512))



        self.res3b_banch2a = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3b_banch2b = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3b_banch2c = nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(512))

        self.res3c_banch2a = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3c_banch2b = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3c_banch2c = nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(512))
        self.res3d_banch2a = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3d_banch2b = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU())
        self.res3d_banch2c = nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(512))  # 7ï¼?0



        self.res4a_banch1 = nn.Sequential(nn.Conv2d(512,1024, kernel_size=5, stride=2, padding=2, dilation=1),
                                          nn.BatchNorm2d(1024))  # 4,15
        self.res4a_banch2a = nn.Sequential(nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=2, dilation=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU())
        self.res4a_banch2b = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4a_banch2c = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(1024))



        self.res4b_banch2a = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4b_banch2b = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4b_banch2c = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(1024))

        self.res4c_banch2a = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4c_banch2b = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4c_banch2c = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(1024))
        self.res4d_banch2a = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4d_banch2b = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4d_banch2c = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(1024))
        self.res4e_banch2a = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4e_banch2b = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.res4e_banch2c = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(1024))  #4,15



        self.res5a_banch1 = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=5, stride=2, padding=2, dilation=1),
                                          nn.BatchNorm2d(2048))  # 2,8
        self.res5a_banch2a = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=5, stride=2, padding=2, dilation=1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU())
        self.res5a_banch2b = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU())
        self.res5a_banch2c = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(2048))




        self.res5b_banch2a = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU())
        self.res5b_banch2b = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU())
        self.res5b_banch2c = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(2048))
        self.res5c_banch2a = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU())
        self.res5c_banch2b = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU())
        self.res5c_banch2c = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, dilation=1),
                                           nn.BatchNorm2d(2048))  # 2,8




        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 1 * 4, 2048),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU())
        self.fc4 = nn.Sequential(
            nn.Linear(500, 6))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):

        conv1_x = self.conv1(x)
        res2a_b1 = self.res2a_banch1(conv1_x)
        res2a_b2 = self.res2a_banch2c(self.res2a_banch2b(self.res2a_banch2a(conv1_x)))
        res2a = F.relu(Eltwise(res2a_b1, res2a_b2))


        res2b_x = self.res2b_banch2c(self.res2b_banch2b(self.res2b_banch2a(res2a)))
        res2b = F.relu(Eltwise(res2a, res2b_x))
        res2c_x = self.res2c_banch2c(self.res2c_banch2b(self.res2c_banch2a(res2b)))
        res2c = F.relu(Eltwise(res2b, res2c_x))
   #####
        res3a_b1 = self.res3a_banch1(res2c)
        res3a_b2 = self.res3a_banch2c(self.res3a_banch2b(self.res3a_banch2a(res2c)))
        res3a = F.relu(Eltwise(res3a_b1, res3a_b2))

        res3b_x = self.res3b_banch2c(self.res3b_banch2b(self.res3b_banch2a(res3a)))
        res3b = F.relu(Eltwise(res3a, res3b_x))
        res3c_x = self.res3c_banch2c(self.res3c_banch2b(self.res3c_banch2a(res3b)))
        res3c = F.relu(Eltwise(res3b, res3c_x))
        res3d_x = self.res3d_banch2c(self.res3d_banch2b(self.res3d_banch2a(res3c)))
        res3d = F.relu(Eltwise(res3c, res3d_x))
    ####
        res4a_b1 = self.res4a_banch1(res3d)
        res4a_b2 = self.res4a_banch2c(self.res4a_banch2b(self.res4a_banch2a(res3d)))
        res4a = F.relu(Eltwise(res4a_b1, res4a_b2))

        res4b_x = self.res4b_banch2c(self.res4b_banch2b(self.res4b_banch2a(res4a)))
        res4b = F.relu(Eltwise(res4a, res4b_x))
        res4c_x = self.res4c_banch2c(self.res4c_banch2b(self.res4c_banch2a(res4b)))
        res4c = F.relu(Eltwise(res4b, res4c_x))
        res4d_x = self.res4d_banch2c(self.res4d_banch2b(self.res4d_banch2a(res4c)))
        res4d = F.relu(Eltwise(res4c, res4d_x))
        res4e_x = self.res4e_banch2c(self.res4e_banch2b(self.res4e_banch2a(res4d)))
        res4e = F.relu(Eltwise(res4d, res4e_x))
      
   #####
        res5a_b1 = self.res5a_banch1(res4e)
        res5a_b2 = self.res5a_banch2c(self.res5a_banch2b(self.res5a_banch2a(res4e)))
        res5a = F.relu(Eltwise(res5a_b1, res5a_b2))

        res5b_x = self.res5b_banch2c(self.res5b_banch2b(self.res5b_banch2a(res5a)))
        res5b = F.relu(Eltwise(res5a, res5b_x))
        res5c_x = self.res5c_banch2c(self.res5c_banch2b(self.res5c_banch2a(res5b)))
        res5c = F.relu(Eltwise(res5b, res5c_x))
   #####
        pool_x = F.avg_pool2d(res5c, kernel_size=2, stride=2)

        last_conv = pool_x.view(pool_x.size()[0], -1)
        result = self.fc4(self.fc3(self.fc2(self.fc1(last_conv))))

        return result


def Eltwise(a, b):
    return a+b



