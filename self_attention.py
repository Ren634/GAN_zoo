import numpy as np
import torch 
from layers import sn_conv2d
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.f = sn_conv2d(in_channels=in_channels,out_channels=in_channels//8,kernel_size=(1,1))
        self.g = sn_conv2d(in_channels=in_channels,out_channels=in_channels//8,kernel_size=(1,1))
        self.h = sn_conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=(1,1))
        self.v = sn_conv2d(in_channels=in_channels//2,out_channels=in_channels,kernel_size=(1,1))
        self.gamma = nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self,x):
        b,c,h,w = x.shape
        fx = self.f(x)
        fx = fx.view(b,c//8,w*h).permute([0,2,1])
        gx = self.g(x)
        gx = torch.max_pool2d(gx,kernel_size=(2,2))
        gx = gx.view(b,c//8,w*h//4)
        attention_map = torch.bmm(fx,gx)
        attention_map = F.softmax(attention_map,dim=-1)
        hx = self.h(x)
        hx = torch.max_pool2d(hx,kernel_size=(2,2))
        hx = hx.view(b,c//2,w*h//4).permute([0,2,1])
        merged_map = torch.bmm(attention_map,hx)
        merged_map = merged_map.permute([0,2,1]).view(b,c//2,h,w)
        attention_map_v = self.v(merged_map)
        return x  + attention_map_v * self.gamma

