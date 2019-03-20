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
import visdom
import argparse

person_set = ["P%02d" % i for i in range(15)]
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--person', choices=person_set, default='P00', help='select test set')
#parser.add_argument('-g', '--gpu', choices=['%d' % i for i in range(4)], default='0', help="choose a gpu")
args = parser.parse_args()
person_num = args.person
print('test person: ', person_num)
try:
    f = open('./record.txt', 'a+')
except IOError:
    f = open('./record.txt', 'w+')
f.write('\n' + person_num)
f.close()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'

H5_address = "/disks/disk0/linyuqi/dataset/data_gaze/h5_txt_list_ALL/"
Save_model_address = '/disks/disk0/linyuqi/model/MPIIGaze_37667/gaze_1280eyes_down/'
BatchSize = 128 
EPOCH = 50
lr_init = 0.0001
count=0
count_ts=0

#vis = visdom.Visdom(env='face_up_train')
#viz = visdom.Visdom(env='face_up_test')

AR_model = model_ns.AR_Net()
AR_model = nn.DataParallel(AR_model)
AR_model.cuda()

E_model = model_ns.E_Net()
E_model = nn.DataParallel(E_model)
E_model.cuda()

#AR_down_model = model_ns.AR_Net_down()
#AR_down_model = nn.DataParallel(AR_down_model)
#AR_down_model.cuda()

#AR_up_model = model_ns.AR_Net_up()
#AR_up_model = nn.DataParallel(AR_down_model)
#AR_up_model.cuda()

loss_f_ARE = loss_func.loss_f_ARE()
loss_f_ARE.cuda()
loss_f_E = loss_func.loss_f_E()
loss_f_E.cuda()

#img_train_list, img_test_list = dataset.load_all_h5(H5_address)

txt_list = os.listdir(H5_address)
txt_list.sort(key=lambda x:int(x[1:3]))
set_dict = dict(zip(person_set, txt_list))
train_list = []
test_list = []
for key in set_dict:
    h5_list = dataset.load_h5_list(os.path.join(H5_address, set_dict[key]))
    if key != person_num:
        train_list.extend(h5_list)
    else:
        test_list.extend(h5_list)
train_Dataset = dataset.gaze_dataset(train_list)
test_Dataset = dataset.gaze_dataset(test_list)
train_loader = torch.utils.data.DataLoader(train_Dataset,shuffle=True,batch_size=BatchSize,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_Dataset,shuffle=True,batch_size=BatchSize,num_workers=4)


L1_loss = nn.SmoothL1Loss().cuda()
#l1_loss = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.01)

optimizer_AR = torch.optim.Adam(AR_model.parameters(),lr=lr_init)
optimizer_E = torch.optim.Adam(E_model.parameters(),lr=lr_init)

time_p, tr_acc,loss_p = [],[],[]

class Min_saver(object):
    def __init__(self):
        self.accu_min = 300
        self.iter = 0
        self.ep = 1

    def set(self, iterarion, accu, epoch):
        self.accu_min = accu
        self.iter = iterarion
        self.ep = epoch
min_saver = Min_saver()  
def angle_error(result_l,result_r,label_l,label_r):
    error_up_l = torch.sum(torch.mul(result_l,label_l),1)        ##torch.Size([128])
    error_d_l = torch.mul(torch.sqrt(torch.sum(torch.pow(result_l,2),1)),     ##torch.Size([128])
                             torch.sqrt(torch.sum(torch.pow(label_l,2),1)))
    angle128_l = torch.acos(torch.clamp((error_up_l/error_d_l),min=-0.999999,max=0.999999))
               

    error_up_r = torch.sum(torch.mul(result_r,label_r),1)
    error_d_r = torch.mul(torch.sqrt(torch.sum(torch.pow(result_r, 2), 1)),
                              torch.sqrt(torch.sum(torch.pow(label_r, 2), 1)))
    angle128_r = torch.acos(torch.clamp((error_up_r / error_d_r),min=-0.999999,max=0.999999))
    return angle128_l, angle128_r
    
def adjust_learning_rate(optimizer, epoch, lr_init):
    if epoch % 35 == 0:
        lr = lr_init * (0.5**(epoch // 25))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print("*********learning_rate change***********")
            print(param_group['lr'])
            
            
def accuracy_text(result_l,result_r,label_l,label_r):
    accuracy_l = 0
    accuracy_r = 0
 
    norm_data_l = torch.sqrt(torch.sum(torch.pow(result_l,2),1))    #[128]
    norm_label_l = torch.sqrt(torch.sum(torch.pow(label_l,2),1))
    angle_value_l = torch.sum(torch.mul(result_l,label_l),1) / (norm_data_l * norm_label_l)
    accuracy_l += (torch.acos(angle_value_l) * 180)/3.1415926     #[128]
        
    norm_data_r = torch.sqrt(torch.sum(torch.pow(result_r,2),1))    #[128]
    norm_label_r = torch.sqrt(torch.sum(torch.pow(label_r,2),1))
    angle_value_r = torch.sum(torch.mul(result_r,label_r),1) / (norm_data_r * norm_label_r)
    accuracy_r += (torch.acos(angle_value_r) * 180)/3.1415926     #[128]
    
    accuracy = (accuracy_l + accuracy_r)/2     #left and right average
    return accuracy

def train(epoch):
    accuracy_all = 0
    global count
    AR_model.train()
    for i ,(two_eyes_img, eye_img_left, eye_img_right, label_left, label_right, head_pose_left, head_pose_right) in enumerate(train_loader):
        start = time.time()
        two_eyes_img = two_eyes_img.cuda()
        eye_img_left = eye_img_left.cuda()
        eye_img_right = eye_img_right.cuda()

        head_pose_left = head_pose_left.cuda()
        head_pose_right = head_pose_right.cuda()
        label_left = label_left.cuda()
        label_right = label_right.cuda()

        count +=1                                   ##1-1563

#        print(label_left)
#        print(head_pose_left)
#        print(label_right)
#        print(head_pose_right)
#        print(image_left)
#        print(image_right)
        optimizer_AR.zero_grad() 
        optimizer_E.zero_grad()

        result_AR = AR_model(two_eyes_img,eye_img_left, eye_img_right,head_pose_left,head_pose_right)       ##output 128 x 6
        result_E = E_model(eye_img_left, eye_img_right)                                        ##output 128 x 2
                             
       # print(result_AR)
       # print(result_E)
        #print(label_left)
        #print(label_right)
        angle128_l, angle128_r = angle_error(result_AR[:,:3],result_AR[:,3:],label_left,label_right)
       
        mat_b = (angle128_l <= angle128_r)
        mat_b = mat_b.float().cuda()
            
        L_E = loss_f_E(result_AR[:,:3].detach(),result_AR[:,3:].detach(),result_E[:,0],result_E[:,1],mat_b.detach())
        omega = (1 + (2 * mat_b - 1) * result_E[:,0] + (1 - 2 * mat_b) * result_E[:,1]) / 2
        L_E.backward()
        optimizer_E.step()
        
        L_ARE = loss_f_ARE(omega.detach(),angle128_l,angle128_r)        
        #L_ARE.register_hook(lambda g:print(g))
        #g = L_ARE.sum()
        #g.backward(retain_graph=True) 

        loss = L_ARE
        loss.backward()
        optimizer_AR.step()        
        
        elapsed = time.time() - start
        
        if count % 23 == 0:
            test_accu = test()
            if test_accu < min_saver.accu_min:
                min_saver.accu_min = test_accu
                min_saver.iter = count
                min_saver.ep = epoch
                
            acc = accuracy_text(result_AR[:,:3],result_AR[:,3:],label_left,label_right)    #two eyes' acc average ([128]) 
            accuracy_avg = acc.mean()                                                   ##one batch acc average  
            #accu = torch.unsqueeze(accuracy_avg,0)          ##([1])

            #x = torch.arange(count,count+1,1)                   ##([1])
            #vis.line(X=x, Y=accu,update = 'append',
            #        win = 'face_tr_accuracy', opts=({'title':'train accuracy and loss'})) 
            print('num_of_iter = {}      min =({}, {}, {:3.3})     train_ARE_loss={}     train_E_loss= {}     time = {:.2f}    acc = {}'.format(
            count, min_saver.ep, min_saver.iter, min_saver.accu_min, loss, L_E, elapsed, accuracy_avg))
        
    
def test():
    accuracy_all = 0
    AR_model.eval()
    global count_ts
    
    length = len(test_loader)
    for i ,(two_eyes_img, eye_img_left, eye_img_right, label_left, label_right, head_pose_left, head_pose_right) in enumerate(test_loader):
        start = time.time()
        two_eyes_img = two_eyes_img.cuda()
        eye_img_left = eye_img_left.cuda()
        eye_img_right = eye_img_right.cuda()

        head_pose_left = head_pose_left.cuda()
        head_pose_right = head_pose_right.cuda()
        label_left = label_left.cuda()
        label_right = label_right.cuda()

        count_ts +=1
#        print(label_left)
#        print(head_pose_left)
#        print(label_right)
#        print(head_pose_right)
        
#        test_AR = AR_model()
#        test_AR.load_state_dict(torch.load(os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(main.epoch))))
#        prediction = test_AR(image_left, image_right,head_pose_left,head_pose_right)

        with torch.no_grad():
           result_AR = AR_model(two_eyes_img,eye_img_left, eye_img_right,head_pose_left,head_pose_right)       ##output 128 x 6
           result_E = E_model(eye_img_left, eye_img_right)                                        ##output 128 x 2                                  
           
#        loss_AR,loss_E,loss_AR2 = loss_f(result_AR[:,:3],result_AR[:,3:],label_left,label_right,result_E[:,0],result_E[:,1])
#        loss = (loss_AR + loss_AR2 +loss_E)
 
        angle128_l, angle128_r = angle_error(result_AR[:,:3],result_AR[:,3:],label_left,label_right)
        #L_AR_ = 2 * torch.mul(angle128_r, angle128_l) / (angle128_r + angle128_l+ 1e-7 )   #+ 1e-4
        #print(angle128_l)
        #print(angle128_r)
        #L_AR = loss_f_AR(angle128_l, angle128_r)
        #L_AR = torch.sum(L_AR_,0)
        #L_AR.backward(retain_graph=True)
        
        mat_b = (angle128_l <= angle128_r)
        mat_b = mat_b.float().cuda()
            
        L_E = loss_f_E(result_AR[:,:3],result_AR[:,3:],result_E[:,0],result_E[:,1],mat_b)
        omega = (1 + (2 * mat_b - 1) * result_E[:,0] + (1 - 2 * mat_b) * result_E[:,1]) / 2
        
        L_ARE = loss_f_ARE(omega,angle128_l,angle128_r)  
        acc = accuracy_text(result_AR[:,:3],result_AR[:,3:],label_left,label_right)       
        accuracy_avg = acc.mean()
        accuracy_all +=accuracy_avg
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
    waste_time = time.time() - start 
    print('test accuracy={:.4} test time {:.3}  '.format(accuracy, waste_time))
    return accuracy
def main():
    for epoch in range(0, EPOCH):
        print('epoch:' + str(epoch) + '\n')
        if train(epoch) == 1:
            f = open('./log.txt', 'a+')
            f.write(person_num + ' error')
            f.close()
            break
        adjust_learning_rate(optimizer_AR, epoch, lr_init)
        adjust_learning_rate(optimizer_E, epoch, lr_init)
        torch.save({'epoch': epoch, 'state_dict': AR_model.state_dict(), }, os.path.join(Save_model_address, 'checkpoint_{}_{}.tar'.format(person_num, epoch)))
    with open('./record.txt', 'r') as f:
        contents = f.read()
        print(contents)
        contents = contents.replace(person_num, person_num + ' %3.3f %d %d' % (min_saver.accu_min, min_saver.iter, min_saver.ep))
        f.close()
    with open('./record.txt', 'w') as f:
        f.write(contents)
        f.close()


if __name__ =='__main__':
    main()       
