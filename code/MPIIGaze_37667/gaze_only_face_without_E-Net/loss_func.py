
import torch
import torch.nn as nn
import torch.nn.functional as func



       
class loss_f_AR(nn.Module):
    def __init__(self):
        super(loss_f_AR, self).__init__()
        return
    def forward(self, angle128_l, angle128_r):
        L_AR_ = 2 * torch.mul(angle128_r, angle128_l) / (angle128_r + angle128_l + 1e-4)  ####torch.Size([128])
        #L_AR_ = (angle128_l-angle128_r)
        L_AR = torch.sum(L_AR_, 0)                              ####scale
        return L_AR
        

