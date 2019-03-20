
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