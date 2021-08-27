import numpy as np
import torch
from torch import nn

def sn_conv2d(**kwargs):
    layer = nn.utils.spectral_norm(nn.Conv2d(**kwargs))
    return layer

def sn_linear(**kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs,))

def sn_tconv2d(**kwargs):
    layer = nn.utils.spectral_norm(nn.ConvTranspose2d(**kwargs))
    return layer

class PixelNorm2d(nn.Module):
    def __init__(self,epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,inputs):
        numerator = torch.mean(inputs**2,dim=0) + self.epsilon
        output = inputs / numerator
        return output

class MiniBatchStddev(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if(torch.cuda.is_available()) else "cpu"

    def forward(self,inputs):
        b,_,h,w = inputs.shape
        std = torch.std(inputs,unbiased=True,dim=0)
        v = torch.mean(std)
        output = torch.cat((inputs,torch.full(size=(b,1,h,w),fill_value=v.item(),device=self.device)),dim=1)
        return output
 