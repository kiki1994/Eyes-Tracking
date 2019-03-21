# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
# face = np.random.randint(1,10,(3,2,2))
# weight = np.random.randint(1,10,[2,2])
# print(face)
# print(weight)                                  # ÑéÖ¤it is written µÄspatial weight
# face_w = np.multiply(face, weight)
# face_w1 = face * weight
# print(face_w)
# print(face_w1)


aa = torch.rand(128, 1, 60, 240)
bb = torch.rand(128, 128, 10, 32)
cc = torch.rand(128, 1, 448, 448)
# bb = torch.rand([128, 1, 3, 32])
# gg = bb.repeat([1,128,1,1])
# dd = torch.ones([128, 1,  3, 32])
# ee = torch.zeros([128, 1,  3, 28])
# ff = torch.zeros([128,128,3,32])
# hh = torch.rand([6,14])
# cc = torch.zeros([6,15])
# print(hh)
# cc[:, :14] = hh
# #cc = cc.view(6,15)
# print(cc)
# dd = dd.unsqueeze(0)
# print(cc)
# print(dd)
# print(cc.shape)           #torch.Size([3, 28])
# print(bb)
# print(gg.shape)

# conv1 = torch.nn.Conv2d(1, 64, 5, 2, 2)
# conv2 = torch.nn.Conv2d(64, 96, 5, 2, 2)   #[1, 3, 15, 60]
# conv3 = torch.nn.Conv2d(96, 128, 5, 2, 2)
# conv4 = torch.nn.Conv2d(128, 256, 5, 2, 2)
# conv5 = torch.nn.Conv2d(256, 384, 5, 2, 2)
# conv6 = torch.nn.Conv2d(384, 256, 3, 1, 1)
# conv7 = torch.nn.Conv2d(256, 256, 3, 1, 0)

conv1 = torch.nn.Conv2d(1, 64, 5, 2, 2)
conv2 = torch.nn.Conv2d(64, 96, 5, 2, 2)   #[1, 3, 15, 60]
conv3 = torch.nn.Conv2d(96, 128, 3, 2, 2)
conv4 = torch.nn.Conv2d(128, 256, 3, 2, 2)
conv5 = torch.nn.Conv2d(256, 384, 3, 1, 1)
conv6 = torch.nn.Conv2d(384, 256, 3, 1, 1)
conv7 = torch.nn.Conv2d(256, 256, 3, 1, 1)
conv8 = torch.nn.Conv2d(256, 256, 3, 1, 0)

pool = torch.nn.MaxPool2d(3, 2)
pool2 = torch.nn.MaxPool2d(2, 2)
result1 = conv1(cc)                      #torch.Size([1, 3, 30, 120])
# result1_p = pool(result1)
result2 = conv2(result1)                 #torch.Size([1, 3, 16, 61])
result3 = conv3(result2)                 #torch.Size([1, 128, 9, 32])

result4 = conv4(result3)                 #torch.Size([1, 3, 12, 32])
# result4_p = pool2(result4)
# result5 = conv5(result4)                 #torch.Size([1, 3, 12, 32])
# result6 = pool(conv6(result5))
# result7 = pool(conv7(result6))
print(result1.shape)
# print(result1_p.shape)
print(result2.shape)
print(result3.shape)
print(result4.shape)
# print(result5.shape)
# print(result6.shape)
# print(result7.shape)
# print(new_tens.shape)

