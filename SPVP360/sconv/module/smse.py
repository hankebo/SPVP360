import torch as th
from torch import nn
from torch.nn import Parameter
from numpy import pi
import math


class SphereMSE(nn.Module):
    def __init__(self, h, w):
        super(SphereMSE, self).__init__()
        self.h, self.w = h, w
        weight = th.zeros(1, 1, h, w)
        theta_range = th.linspace(0, pi, steps=h + 1)
        dtheta = pi / h
        dphi = 2 * pi / w
        for theta_idx in range(h):
            weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[theta_idx]) + math.sin(theta_range[theta_idx+1]))/2 * dtheta
        self.weight = Parameter(weight, requires_grad=False)

    def forward(self, out, target):
        return th.sum((out - target) ** 2 * self.weight) / out.size(0)
    
    
    
class SFLoss(nn.Module):
    def __init__(self,  h, w, beta=0.3,log_like=False):
        super(SFLoss, self).__init__()
        self.h, self.w = h, w
        weight = th.zeros(1, 1, h, w)
        theta_range = th.linspace(0, pi, steps=h + 1)
        dtheta = pi / h
        dphi = 2 * pi / w
        for theta_idx in range(h):
            weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[theta_idx]) + math.sin(theta_range[theta_idx+1]))/2 * dtheta
        self.weight = Parameter(weight, requires_grad=False)
        
        self.beta = beta
        self.log_like = log_like

    def forward(self, out, target):
        EPS = 1e-10
        N = out.size(0)
        TP = (out * target* self.weight ).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + out.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -th.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss


    