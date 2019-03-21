
import torch
import torch.nn as nn
import torch.nn.functional as func



class loss_f_ARE(nn.Module):
    def __init__(self):
        super(loss_f_ARE, self).__init__()
        return
    def forward(self, omega,angle128_l,angle128_r):
#        L_E128 = -(mat_b * torch.mul(torch.acos(L_E128_up/L_E128_down),torch.log(prob_l)) + (1 - mat_b) * torch.mul(torch.acos(L_E128_up/L_E128_down),(torch.log(prob_r))))
        L_AR_ = 2 * torch.mul(angle128_r, angle128_l) / (angle128_r + angle128_l+ 1e-7 )
        L_AR2_ = torch.mul(omega, L_AR_) + 0.1*torch.mul((1 - omega), ((angle128_r + angle128_l) / 2))  ### The new one
        L_AR2 = torch.sum(L_AR2_,0)
         
        return L_AR2
        
class loss_f_AR(nn.Module):
    def __init__(self):
        super(loss_f_AR, self).__init__()
        return
    def forward(self, angle128_l, angle128_r):
        L_AR_ = 2 * torch.mul(angle128_r, angle128_l) / (angle128_r + angle128_l + 1e-4)  ####torch.Size([128])
        #L_AR_ = (angle128_l-angle128_r)
        L_AR = torch.sum(L_AR_, 0)                              ####scale
        return L_AR
        
class loss_f_E(nn.Module):
    def __init__(self):
        super(loss_f_E, self).__init__()
        return
    def forward(self, result_l, result_r,prob_l,prob_r,mat_b):
        L_E128_up = torch.sum(torch.mul(result_l, result_r), 1)  ##torch.Size([128])
        L_E128_down = torch.mul(torch.sqrt(torch.sum(torch.pow(result_l, 2), 1)),  ##torch.Size([128])
                            torch.sqrt(torch.sum(torch.pow(result_r, 2), 1)))
        L_E128 = -(mat_b *torch.acos(torch.clamp((L_E128_up/L_E128_down),min=-1,max=1))*torch.log(torch.clamp(prob_l,min=1e-30,max=0.99999999)) +(1 - mat_b) * torch.acos(torch.clamp((L_E128_up/L_E128_down),min=-1,max=1))*(torch.log(torch.clamp(prob_r,min=1e-30,max=0.99999999))))
        #L_E128 = -(mat_b*torch.acos(L_E128_up)*torch.log(prob_l)+(1-mat_b)*torch.acos(L_E128_up)*torch.log(prob_r))
        L_E = torch.sum(L_E128, 0) 
        return  L_E
        
     
