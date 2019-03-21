# -*- coding: utf-8 -*-

import dataset
import model_ns
import loss_func

import torch
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import itertools
#import visdom
import argparse

person_set = ["P%02d" % i for i in range(15)]
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--person', choices=person_set, default='P00', help='select test set')
parser.add_argument('-g', '--gpu', choices=['%d' % i for i in range(4)], default='0', help="choose a gpu")
args = parser.parse_args()
person_num = args.person
print('test person: ', person_num)
try:
    f = open('./record.txt', 'a+')
except IOError:
    f = open('./record.txt', 'w+')
f.write('\n' + person_num)
f.close()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#H5_address = "/4Tdisk/lyq/create_h5/h5_alldata/h5_txtlist/"
H5_address =  "/disks/disk0/linyuqi/dataset/data_gaze/h5_dataset/h5_loc_grey_37667/h5_txtlist/"
#Save_model_address = "/disks/disk0/linyuqi/model/MPIIGaze_37667/448face_1280model_up/"
BatchSize = 128 
EPOCH = 100
lr_init = 0.0001
count=0
count_ts=0


model = model_ns.BaseMode()
model = nn.DataParallel(model)
model.cuda()

#JSloss = loss_func.JS()
#JSloss.cuda()

ARCCloss = loss_func.ARCC()
ARCCloss.cuda()

txt_list = os.listdir(H5_address)
txt_list.sort(key=lambda x:int(x[1:3]))     #key = 00, 01, ...14
set_dict = dict(zip(person_set, txt_list))
train_list = []
test_list = []
for key in set_dict:
    h5_list = dataset.load_h5_list(os.path.join(H5_address, set_dict[key]))
    #print(h5_list)
    #print(len(h5_list))
    if key != person_num:
        train_list.extend(h5_list)
#        print(len(train_list))
#        print(train_list)
    else:
        test_list.extend(h5_list)
#        print(len(test_list))
#        print(test_list)
train_Dataset = dataset.gaze_dataset(train_list)
test_Dataset = dataset.gaze_dataset(test_list)
train_loader = torch.utils.data.DataLoader(train_Dataset,shuffle=True,batch_size=BatchSize,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_Dataset,shuffle=True,batch_size=BatchSize,num_workers=4)

MSEloss = nn.MSELoss(size_average = True).cuda()
#L1_loss = nn.SmoothL1Loss().cuda()
#l1_loss = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.01)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

time_p, tr_acc,loss_p = [],[],[]

class Min_saver(object):
    def __init__(self):
        self.accu_min = 300
        self.iter = 0
        self.ep = 0
        self.accu_min_l = 300
        self.accu_min_r = 300
        self.train_loss = 300

    def set(self, iterarion, accu, epoch, accu_l, accu_r, train_loss):
        self.accu_min = accu
        self.iter = iterarion
        self.ep = epoch
        self.accu_min_l = accu_l
        self.accu_min_r = accu_r
        self.train_loss = train_loss

min_saver = Min_saver()
  
# def angle_error(result_l,result_r,label_l,label_r):
#     error_up_l = torch.sum(torch.mul(result_l,label_l),1)        ##torch.Size([128])
#     error_d_l = torch.mul(torch.sqrt(torch.sum(torch.pow(result_l,2),1)),     ##torch.Size([128])
#                              torch.sqrt(torch.sum(torch.pow(label_l,2),1)))
#     angle128_l = torch.acos(torch.clamp((error_up_l/error_d_l),min=-0.999999,max=0.999999))
#
#
#     error_up_r = torch.sum(torch.mul(result_r,label_r),1)
#     error_d_r = torch.mul(torch.sqrt(torch.sum(torch.pow(result_r, 2), 1)),
#                               torch.sqrt(torch.sum(torch.pow(label_r, 2), 1)))
#     angle128_r = torch.acos(torch.clamp((error_up_r / error_d_r),min=-0.999999,max=0.999999))
#     return angle128_l, angle128_r
#
def adjust_learning_rate(optimizer, epoch, lr_init):
    if epoch % 35 == 0:
        lr = lr_init * (0.5**(epoch // 25))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print("*********learning_rate change***********")
            print(param_group['lr'])


def accuracy_text(result_l, result_r, label_l, label_r):
    accuracy_l = 0
    accuracy_r = 0

    norm_data_l = torch.sqrt(torch.sum(torch.pow(result_l, 2), 1))  # [128]
    norm_label_l = torch.sqrt(torch.sum(torch.pow(label_l, 2), 1))
    angle_value_l = torch.sum(torch.mul(result_l, label_l), 1) / (norm_data_l * norm_label_l)
    accuracy_l += (torch.acos(angle_value_l) * 180) / 3.1415926  # [128]

    norm_data_r = torch.sqrt(torch.sum(torch.pow(result_r, 2), 1))  # [128]
    norm_label_r = torch.sqrt(torch.sum(torch.pow(label_r, 2), 1))
    angle_value_r = torch.sum(torch.mul(result_r, label_r), 1) / (norm_data_r * norm_label_r)
    accuracy_r += (torch.acos(angle_value_r) * 180) / 3.1415926  # [128]

    accuracy = (accuracy_l + accuracy_r) / 2  # left and right average
    return accuracy, accuracy_l, accuracy_r

def train(epoch):
    accuracy_all = 0
    global count
    model.train()
    file_path = './epoch_acc_{}.txt'.format(person_num)
    with open(file_path, 'a+') as file:
        for i, (two_eyes_img, label_left, label_right, head_pose_left, head_pose_right, eye_loc, face_loc) in enumerate(train_loader):
            start = time.time()
            count +=1

            two_eyes_img = two_eyes_img.cuda()

            head_pose_left = head_pose_left.cuda()
            head_pose_right = head_pose_right.cuda()
            label_left = label_left.cuda()
            label_right = label_right.cuda()
            eye_loc = eye_loc.cuda()
            face_loc = face_loc.cuda()

            optimizer.zero_grad()
            result = model(two_eyes_img, eye_loc, face_loc, head_pose_left, head_pose_right)       ##output 128 x 6
            result_l = result[:, :3]
            result_r = result[:, 3:]
            #label = torch.cat([label_left, label_right], 1)
            #loss = JSloss(result, label)
            #loss = WJSloss(label,result)
            #loss = MSEloss(result, label)
            loss = ARCCloss(result_l, result_r, label_left, label_right)
            loss.backward()
            result.grad
            optimizer.step()

            elapsed = time.time() - start

            if count % 23 == 0:
                test_accu, test_accu_l, test_accu_r = test()
                iter_data = '{}-{}: {:3.3} {:3.3} {:3.3}'.format(epoch, count, test_accu, test_accu_l, test_accu_r)
                if test_accu < min_saver.accu_min:
                    min_saver.accu_min = test_accu
                    min_saver.iter = count
                    min_saver.ep = epoch
                if test_accu_l < min_saver.accu_min_l:
                    min_saver.accu_min_l = test_accu_l
                if test_accu_r < min_saver.accu_min_r:
                    min_saver.accu_min_r = test_accu_r
                if loss < min_saver.train_loss:
                    min_saver.train_loss = loss

                acc, acc_l, acc_r = accuracy_text(result_l, result_r, label_left, label_right)    #two eyes' acc average ([128])
                accuracy_avg = acc.mean()                                                   ##one batch acc average
                #accu = torch.unsqueeze(accuracy_avg,0)          ##([1])

                #x = torch.arange(count,count+1,1)                   ##([1])
                #vis.line(X=x, Y=accu,update = 'append',
                #        win = 'face_tr_accuracy', opts=({'title':'train accuracy and loss'}))
                print('test_min_accu =({}, {:3.3})    test_min_eyes_accu =({:3.3},{:3.3})  train_loss = {}     time = {}   train_accu = {}'.format(
                min_saver.iter, min_saver.accu_min, min_saver.accu_min_l, min_saver.accu_min_r, loss, elapsed, accuracy_avg))
                file.write('\n' + iter_data)
    file.close()
    
def test():
    accuracy_all = 0
    accuracy_left_all = 0
    accuracy_right_all = 0
    model.eval()
    global count_ts
    length = len(test_loader)
    for i, (two_eyes_img, label_left, label_right, head_pose_left, head_pose_right, eye_loc, face_loc) in enumerate(test_loader):
        start = time.time()

        two_eyes_img = two_eyes_img.cuda()
        head_pose_left = head_pose_left.cuda()
        head_pose_right = head_pose_right.cuda()
        label_left = label_left.cuda()
        label_right = label_right.cuda()
        eye_loc = eye_loc.cuda()
        face_loc = face_loc.cuda()

        count_ts +=1

        
#        test_AR = AR_model()
#        test_AR.load_state_dict(torch.load(os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(main.epoch))))
#        prediction = test_AR(image_left, image_right,head_pose_left,head_pose_right)

        with torch.no_grad():
           result = model(two_eyes_img, eye_loc, face_loc, head_pose_left, head_pose_right)       ##output 128 x 6


        result_l = result[:, :3]
        result_r = result[:, 3:]

        label = torch.cat([label_left, label_right], 1)
        #loss = JSloss(result, label)
        #loss = WJSloss(label, result)
        #loss = MSEloss(result, label)
        loss = ARCCloss(result_l, result_r, label_left, label_right)

        acc, acc_l, acc_r = accuracy_text(result_l, result_r, label_left, label_right)
        accuracy_avg = acc.mean()
        accuracy_left_avg = acc_l.mean()
        accuracy_right_avg = acc_r.mean()

        accuracy_all += accuracy_avg
        accuracy_left_all += accuracy_left_avg
        accuracy_right_all += accuracy_right_avg
        #print(accuracy) 
        #print("-----test-----")
        #print(count_ts)
               
             
            #accu = torch.unsqueeze(accuracy_avg,0)          ##([1]) 
            #loss_are = torch.unsqueeze(L_ARE,0)
            #loss_e = torch.unsqueeze(L_E,0)
         
            #x = torch.arange(count_ts,count_ts+1,1)                   ##([1])
            
            #viz.line(X=np.column_stack((x,x,x)), Y=torch.from_numpy(np.column_stack((accu,loss_e,loss_are))),update = 'append',
            #         win = 'face_ts_accuracy and loss',opts=dict({'title':'test accuracy and loss'},legend=["acc","L_E","L_ARE"]))  
                     
            #print('num_of_batch = {}    test_loss={}    accuracy={}   '.format(i,L_ARE,accuracy_avg))
            
            #print('left predit:({:5.4},{:5.4},{:5.4})   right predit:({:5.4},{:5.4},{:5.4})'.format(result_AR[i][0],result_AR[i][1],result_AR[i][2],result_AR[i][3],result_AR[i][4],result_AR[i][5]))

            #print('label_left:({:5.4},{:5.4},{:5.4})     label_right:({:5.4},{:5.4},{:5.4})'.format(label_left[i][0],label_left[i][1],label_left[i][2],label_right[i][0],label_right[i][1],label_right[i][2]))
   
    accuracy = accuracy_all/length
    accuracy_l = accuracy_left_all/length
    accuracy_r = accuracy_right_all/length
    waste_time = time.time() - start
    print('test accu ={:.4}, test_eyes_accu = ({:.4},{:.4}), test time = {:.3}  '.format(accuracy, accuracy_l, accuracy_r, waste_time))
    print('==========================================================')
    return accuracy, accuracy_l, accuracy_r

def main():
    for epoch in range(0, EPOCH):
        print('epoch:' + str(epoch) + '\n')
        if train(epoch) == 1:
            f = open('./log.txt', 'a+')
            f.write(person_num + ' error')
            f.close()
            break
         #torch.save({'epoch': epoch,
                    #'state_dict': AR_model.state_dict(), }, os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(epoch)))
        adjust_learning_rate(optimizer, epoch, lr_init)
    with open('./record.txt', 'r') as f:
        contents = f.read()
        print(contents)
        contents = contents.replace(person_num, person_num + ' %3.3f %d %3.3f %3.3f' % (min_saver.accu_min, min_saver.ep, min_saver.accu_min_l, min_saver.accu_min_r))
        f.close()
    with open('./record.txt', 'w') as f:
        f.write(contents)
        f.close()


if __name__ =='__main__':
    main()       
