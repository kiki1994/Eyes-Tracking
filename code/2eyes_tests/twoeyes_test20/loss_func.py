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
        JSD = 0.5 * (real_v * t.log(real_v/M)+( pre_v * t.log( pre_v/M)))
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
        
class WJS(nn.Module):
    def __int__(self):
        super(WJS, self).__int__()
        return
    def forward(self, _p, _q):
        p = func.softmax(_p,1)
        q = func.softmax(_q,1)
        row = p.size()[0]
        line = p.size()[1]
        jsd = t.zeros([row, line]).cuda()
        reg_jsd = t.zeros([row, line]).cuda()
        m = 0.5 * (p + q)
        for i in range(row):
            for j in range(line):
                jsd[i][j] = 0.5*(q[i][j] * t.log(q[i][j] / m[i][j])+ p[i][j] * t.log(p[i][j] / m[i][j]))
        sumjsd = t.sum(jsd, 1)
#        for i in range(row):
#            for j in range(line):
#                reg_jsd[i][j] = jsd[i][j] / sumjsd[i]  
        weuc = t.sum((t.sqrt(t.sum((t.mul(t.pow((_p-_q), 2), jsd)),1))),0)                         
        return weuc
        
class WEUCD(nn.Module):
    def __int__(self):
        super(WEUCD, self).__int__()
        return
    def forward(self, p, q):   
        weuc = t.sum((t.sqrt(t.sum((t.mul(t.pow((p-q), 2), reg_jsd)),1))),0)
        return weuc