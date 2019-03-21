import torch as t
import torch
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
        
class ARCC(nn.Module):
    def __int__(self):
        super(ARCC, self).__int__()
        return
    def forward(self, result_l, result_r, label_l, label_r):
        accuracy_l = t.zeros([result_l.size()[0]]).cuda()
        accuracy_r = t.zeros([result_l.size()[0]]).cuda()

        norm_data_l = torch.sqrt(torch.sum(torch.pow(result_l, 2), 1))  # [128]
        norm_label_l = torch.sqrt(torch.sum(torch.pow(label_l, 2), 1))
        angle_value_l = torch.sum(torch.mul(result_l, label_l), 1) / (norm_data_l * norm_label_l)
        accuracy_l += (torch.acos(angle_value_l) * 180) / 3.1415926  # [128]

        norm_data_r = torch.sqrt(torch.sum(torch.pow(result_r, 2), 1))  # [128]
        norm_label_r = torch.sqrt(torch.sum(torch.pow(label_r, 2), 1))
        angle_value_r = torch.sum(torch.mul(result_r, label_r), 1) / (norm_data_r * norm_label_r)
        accuracy_r += (torch.acos(angle_value_r) * 180) / 3.1415926  # [128]

        accuracy = t.sum(((accuracy_l + accuracy_r) / 2),0)  # left and right average
        return accuracy