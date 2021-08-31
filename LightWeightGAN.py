import numpy as np
import torch
from gan_modules import * 
from torch import nn
from torch.nn import functional as F
from math import log2
from collections import deque
class SLE(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            sn_conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(4,4),stride=1,padding=0),
            nn.LeakyReLU(0.1),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x,shutcut):
        x = self.main(x)
        output = x * shutcut
        return output
        
class BlockG(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=(2,2),mode="nearest"),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels*2,kernel_size=(3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.GLU(dim=1),
        )
        
    def forward(self,x):
        output = self.main(x)
        return output

class Generator(nn.Module):
    def __init__(self,n_dims,max_resolution):
        super().__init__()
        out_channels = {
            4:256,
            8:512,
            16:512,
            32:256,
            64:128,
            128:64,
            256:32,
            512:3,
            1024:3
        }
        self.max_resolution = max_resolution
        self.feat128 = nn.ModuleList([
            nn.Sequential(
                sn_tconv2d(in_channels=n_dims,out_channels=out_channels[4]*2,kernel_size=(4,4),stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_channels[4]*2),
                nn.GLU(dim=1)
            ),
            BlockG(in_channels=out_channels[4],out_channels=out_channels[8]),
            BlockG(in_channels=out_channels[8],out_channels=out_channels[16]),
            BlockG(in_channels=out_channels[16],out_channels=out_channels[32]),
            BlockG(in_channels=out_channels[32],out_channels=out_channels[64]),
            BlockG(in_channels=out_channels[64],out_channels=out_channels[128]),            
        ])
        self.feat128_to_max = nn.ModuleList()
        self.SLE_layer = nn.ModuleList([SLE(in_channels=out_channels[8],out_channels=out_channels[128])])
        for index in range(7,int(log2(max_resolution))):
            self.feat128_to_max.append(BlockG(in_channels=out_channels[2**(index)],out_channels=out_channels[2**(index+1)])) 
            if(2**(index)!=512):
                self.SLE_layer.insert(0,SLE(in_channels=out_channels[2**(index+1-4)],out_channels=out_channels[2**(index+1)]))
                        
        self.output_layer = nn.Sequential(
            sn_conv2d(in_channels=out_channels[max_resolution],out_channels=3,kernel_size=(3,3),stride=1,padding=1),
            nn.Tanh()
        )
        self.intermidiate = deque()

    def forward(self,x):
        for layer in self.feat128:
            x = layer(x)
            if(8<=x.shape[-1]<=2**(int(log2(self.max_resolution))-4)):
                self.intermidiate.append(x)
        for layer in self.feat128_to_max:
            if(len(self.intermidiate)>0):
                x = self.SLE_layer[len(self.intermidiate)-1](self.intermidiate.popleft(),x)
            x = layer(x)
        output = self.output_layer(x)
        return output
