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
                                   nn.BatchNorm2d(64),   #224
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=5, stride=2, padding=2, dilation=1),
                                   nn.BatchNorm2d(96),  # 112
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=2, dilation=1),
                                   nn.BatchNorm2d(128),  # 57
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2, dilation=1),
                                   nn.BatchNorm2d(256),  # 30
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(512),  # 30
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(512),  # 30
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, stride=2)) #14
        self.conv7 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, stride=2))    # 6
        self.conv8 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, stride=2))    # 1
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, 2048),
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

    def forward(self, x, eyes_loc, face_loc, left_headpose, right_headpose):
        add_points = torch.zeros([x.size()[0], 256, 3, 30]).cuda()
        points = torch.cat([eyes_loc, face_loc, left_headpose, right_headpose], 1)
        #print(points.shape)
        points = points.view(x.size()[0], 3, 28)                     # torch.Size([128, 3, 28])
        points = points.unsqueeze(1)
        #print(points.shape)
        points = points.repeat([1, 256, 1, 1])                   #[128, 128, 3, 28]     
        #print(points.shape)                            
        add_points[:, :, :, :28] = points
        #print(add_points.shape)

        x_img = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        #print(x_img.shape)
        new_tens = torch.cat([x_img, add_points], 2)                                 #torch.Size([1, 128, 12, 32])
        last_conv = self.conv8(self.conv7(self.conv6(self.conv5(new_tens))))         #torch.Size([1, 256, 1, 6])
        last_conv = last_conv.view(last_conv.size()[0], -1)
        result = self.fc4(self.fc3(self.fc2(self.fc1(last_conv))))

        return result



def Spatialweights(facemap, fcface):
    weightmap = torch.zeros([128, 256, 12, 12]).cuda()
    weightmap.requries_grad = True
    #    print(facemap.shape[1])
    for i in range(facemap.shape[1]):
        weightmap[0, i, :, :] = torch.mul(facemap[0, i, :, :], fcface[0, 0, :, :])
    # print(facemap[0,i,:,:])            #(1, 256, 13, 13)
    #    print(weightmap.shape)
    return weightmap


