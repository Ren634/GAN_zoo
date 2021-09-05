import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from math import sqrt

def sn_conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True,**kwargs):
    layer = nn.utils.spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            **kwargs)
        ) 
    return layer

def sn_linear(in_features,out_features,bias=True,**kwargs):
    return nn.utils.spectral_norm(nn.Linear(in_features=in_features,out_features=out_features,bias=bias,**kwargs))

def sn_tconv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True,**kwargs):
    layer = nn.utils.spectral_norm(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
            **kwargs)
        )
    return layer

class EqualizedLRTConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True,**kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs
        )
        nn.init.normal_(self.conv.weight,mean=0,std=1)
        self.scale_factor = sqrt(2) / in_channels

    def forward(self,inputs):
        with torch.no_grad():
            self.conv.weight = self.conv / self.scale_factor
        output = self.conv(inputs)
        return output

class EqualizedLRConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True,**kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs
        )
        nn.init.normal_(self.conv.weight,mean=0,std=1)
        self.scale_factor = sqrt(2) / in_channels

    def forward(self,inputs):
        with torch.no_grad():
            self.conv.weight = self.conv / self.scale_factor
        output = self.conv(inputs)
        return output  
    
class EqualizedLRLinear(nn.Module):
    def __init__(self,in_features,out_features,bias=True,**kwargs):
        super().__init__()
        self.conv = nn.Linear(in_features=in_features,out_features=out_features,bias=bias)        
        nn.init.normal_(self.conv.weight,mean=0,std=1)
        self.scale_factor = sqrt(2) / in_features

    def forward(self,inputs):
        with torch.no_grad():
            self.conv.weight = self.conv / self.scale_factor
        output = self.conv(inputs)
        return output

class PixelNorm2d(nn.Module):
    def __init__(self,epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,inputs):
        denominator = torch.mean(inputs**2,dim=0) + self.epsilon
        output = inputs / denominator
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

class GlobalSum(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self,inputs):
        return torch.sum(inputs,dim=(-1,-2))

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


 
