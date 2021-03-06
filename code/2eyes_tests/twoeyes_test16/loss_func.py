import torch as t
import torch.nn as nn
import torch.nn.functional as func



class JS(nn.Module):
    def __init__(self):
        super(JS, self).__init__()
        return
    def forward(self, pre_v, real_v):
#              
        M = 0.5 *(pre_v + real_v)
        JSD = 0.5 * (real_v * torch.log(torch.clamp(real_v/M, min=1e-30,max=0.99999999))+( pre_v * torch.log(torch.clamp( pre_v/M, min=1e-30,max=0.99999999))))
        return JSD


class EUCD(nn.Module):
    def __int__(self):
        super(EUCD, self).__int__()
        return
    def forward(self, prel, prer, reall, realr):
        dl = t.sqrt(t.sum(t.pow((reall - prel), 2), 1))
        dr = t.sqrt(t.sum(t.pow((realr - prer), 2), 1))
        d = t.sum((dl+dr),0)
        return d   
        
class WEUCD(nn.Module):
    def __int__(self):
        super(WEUCD , self).__int__()
        return
    def forward(self, _p, _q):
        weight = t.Tensor([0.1,0.1,0.3,0.1,0.1,0.3]).cuda()
        weight = weight.unsqueeze(0)
        weight = weight.repeat([_p.size()[0],1])
        weuc = t.sum((t.sqrt(t.sum((t.mul(t.pow((_p-_q), 2), weight)),1))),0)   
        return weuc