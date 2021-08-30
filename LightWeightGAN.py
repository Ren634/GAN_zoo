import numpy as np
import torch
from gan_modules import *
from torch import nn
from torch.nn import functional as F

class SLE(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4),stride=1,padding=0),
            nn.LeakyReLU(0.1),
            sn_conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x,shutcut):
        x = self.main(x)
        output = x * shutcut
        return output
        
class BlockG(nn.Module):
    def __init__(self,in_channels,out_channles):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=(2,2),mode="nearest"),
            sn_conv2d(in_channels=in_channels,out_channels=out_channles*2,kernel_size=(3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channles*2),
            nn.GLU(dim=1),
        )
        
    def forward(self,x):
        output = self.main(x)
        return output
