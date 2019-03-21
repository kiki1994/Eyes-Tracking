# -*- coding: utf-8 -*-



import torch
import dataset
import model_ns
import loss_func

import os
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
import visdom


H5_address = '/disks/disk0/linyuqi/dataset/data_gaze/gaze_detection/h5_2eyes/'
Save_model_address = '/disks/disk0/linyuqi/model/gaze_1280eye_up/'
BatchSize = 128 
EPOCH = 30
vis = visdom.Visdom(env='full_face_AR_up')
viz = visdom.Visdom(env='full_face_AR_up_test')
count=0
count_ts=0
train_iteration =np.ceil(200000/BatchSize)
test_iteration = np.ceil(13659/BatchSize)

AR_model = model_ns.AR_Net()
AR_model = nn.DataParallel(AR_model)
AR_model.cuda()

E_model = model_ns.E_Net()
E_model = nn.DataParallel(E_model)
E_model.cuda()

AR_down_model = model_ns.AR_Net_down()
AR_down_model = nn.DataParallel(AR_down_model)
AR_down_model.cuda()

AR_up_model = model_ns.AR_Net_up()
AR_up_model = nn.DataParallel(AR_down_model)
AR_up_model.cuda()

loss_f_ARE = loss_func.loss_f_ARE()
loss_f_ARE.cuda()
loss_f_E = loss_func.loss_f_E()
loss_f_E.cuda()


img_name_list = dataset.load_all_h5(H5_address)
#print(len(img_list_left))
#print(len(img_list_right))
train_Dataset = dataset.train_gaze_dataset(img_name_list)
test_Dataset = dataset.test_gaze_dataset(img_name_list)
train_loader = torch.utils.data.DataLoader(train_Dataset,shuffle=True,batch_size=BatchSize,num_workers=6)
test_loader = torch.utils.data.DataLoader(test_Dataset,shuffle=True,batch_size=BatchSize,num_workers=6)


L1_loss = nn.SmoothL1Loss().cuda()
#l1_loss = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(gaze_model.parameters(),lr=0.01)

optimizer_AR = torch.optim.SGD(AR_model.parameters(),lr=0.0001,momentum=0.9)
optimizer_E = torch.optim.SGD(E_model.parameters(),lr=0.0001,momentum=0.9)

time_p, tr_acc,loss_p = [],[],[]
   
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

def d_3(result):
    data = torch.zeros([result.size()[0], 3]) 
        
    for i in range(0, result.size()[0]):

        data[i][0] = (-1) * (torch.cos(result[i][0])) * (torch.sin(result[i][1]))

        data[i][1] = (-1) * (torch.sin(result[i][0]))

        data[i][2] = (-1) * (torch.cos(result[i][0])) * (torch.cos(result[i][1]))

    #tens_3 = torch.cat([data[:, 0], data[:, 1], data[:, 2]], 0)
    #tens1_3 = torch.unsqueeze(tens_3, 0)

    return data   ##size 128*3
    
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



def train():
    accuracy_all = 0
    global count
    AR_model.train()
    start = time.time()
    for i ,(face_img, eye_img_left, eye_img_right, label_left, label_right, head_pose_left, head_pose_right) in enumerate(train_loader):
        face_img = face_img.squeeze(1)
        face_img = face_img.cuda()
        eye_img_left = eye_img_left.squeeze(1)
        eye_img_left = eye_img_left.cuda()
        eye_img_right = eye_img_right.squeeze(1)
        eye_img_right = eye_img_right.cuda()

        head_pose_left = head_pose_left.cuda()
        head_pose_right = head_pose_right.cuda()
        label_left = label_left.cuda()
        label_right = label_right.cuda()

#Turn to 3D
        label_left = d_3(label_left).cuda()
        label_right = d_3(label_right).cuda()
        head_pose_left = d_3(head_pose_left).cuda()
        head_pose_right = d_3(head_pose_right).cuda()

        count +=1                                   ##1-1563

#        print(label_left)
#        print(head_pose_left)
#        print(label_right)
#        print(head_pose_right)
#        print(image_left)
#        print(image_right)
        optimizer_AR.zero_grad() 
        optimizer_E.zero_grad()

        result_AR = AR_model(face_img,eye_img_left, eye_img_right,head_pose_left,head_pose_right)       ##output 128 x 6
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
                
        if i%20 ==0:  
            acc = accuracy_text(result_AR[:,:3],result_AR[:,3:],label_left,label_right)    #two eyes' acc average ([128]) 
            accuracy_avg = acc.mean()                                                   ##one batch acc average  
            accu = torch.unsqueeze(accuracy_avg,0)          ##([1])

            x = torch.arange(count,count+1,1)                   ##([1])
            vis.line(X=x, Y=accu,update = 'append',
                    win = 'face_tr_accuracy', opts=({'title':'train accuracy and loss'})) 
       # if i%80 ==0:
            print('num_of_batch = {}    train_loss={}      time = {:.2f}       acc = {}'.format(i,loss,elapsed,accuracy_avg))
        
    
def test():
    accuracy_all = 0
    AR_model.eval()
    global count_ts
    for i ,(face_img, eye_img_left, eye_img_right, label_left, label_right, head_pose_left, head_pose_right) in enumerate(test_loader):
        face_img = face_img.squeeze(1)
        face_img = face_img.cuda()
        eye_img_left = eye_img_left.squeeze(1)
        eye_img_left = eye_img_left.cuda()
        eye_img_right = eye_img_right.squeeze(1)
        eye_img_right = eye_img_right.cuda()

        head_pose_left = head_pose_left.cuda()
        head_pose_right = head_pose_right.cuda()
        label_left = label_left.cuda()
        label_right = label_right.cuda()

#Turn to 3D
        label_left = d_3(label_left).cuda()
        label_right = d_3(label_right).cuda()
        head_pose_left = d_3(head_pose_left).cuda()
        head_pose_right = d_3(head_pose_right).cuda()

        count_ts +=1
#        print(label_left)
#        print(head_pose_left)
#        print(label_right)
#        print(head_pose_right)
        
#        test_AR = AR_model()
#        test_AR.load_state_dict(torch.load(os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(main.epoch))))
#        prediction = test_AR(image_left, image_right,head_pose_left,head_pose_right)

        with torch.no_grad():
           result_AR = AR_model(face_img,eye_img_left, eye_img_right,head_pose_left,head_pose_right)       ##output 128 x 6
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
               
        if i%20 ==0:      
            accu = torch.unsqueeze(accuracy_avg,0)          ##([1]) 
            loss_are = torch.unsqueeze(L_ARE,0)
            loss_e = torch.unsqueeze(L_E,0)
         
            x = torch.arange(count_ts,count_ts+1,1)                   ##([1])
            
            viz.line(X=np.column_stack((x,x,x)), Y=torch.from_numpy(np.column_stack((accu,loss_e,loss_are))),update = 'append',
                     win = 'face_ts_accuracy and loss',opts=dict({'title':'test accuracy and loss'},legend=["acc","L_E","L_ARE"]))  
                     
            #print('num_of_batch = {}    test_loss={}    accuracy={}   '.format(i,L_ARE,accuracy_avg))
            
            #print('left predit:({:5.4},{:5.4},{:5.4})   right predit:({:5.4},{:5.4},{:5.4})'.format(result_AR[i][0],result_AR[i][1],result_AR[i][2],result_AR[i][3],result_AR[i][4],result_AR[i][5]))

            #print('label_left:({:5.4},{:5.4},{:5.4})     label_right:({:5.4},{:5.4},{:5.4})'.format(label_left[i][0],label_left[i][1],label_left[i][2],label_right[i][0],label_right[i][1],label_right[i][2]))

        
    
    accuracy = accuracy_all/test_iteration
    print(accuracy) 
    print("-----test-----")

def main():

    for epoch in range(0,EPOCH):
        print('epoch:' + str(epoch) + '\n')
        train()
        test()
        torch.save({'epoch':epoch,
                    'state_dict':AR_model.state_dict(),},os.path.join(Save_model_address, 'checkpoint_{}.tar'.format(epoch)))
        # save only the parameters
#        test()


if __name__ =='__main__':
    main()       
