import torch
from torch.nn import functional as F
from torch import nn
from gan_modules import *
from math import log2

class BlockG(nn.Module):
    def __init__(self,in_chanels,out_channels,is_input_layer=False):
        super().__init__()
        if(is_input_layer):
            ConvT2d = nn.ConvTranspose2d(in_channels=in_chanels,out_channels=out_channels,kernel_size=(4,4),bias=False)
        else:
            ConvT2d = nn.ConvTranspose2d(in_channels=in_chanels,out_channels=out_channels,kernel_size=(2,2),stride=2,bias=False)
        self.main = nn.Sequential(
            ConvT2d,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True) 
        )

    def forward(self,x):
        output = self.main(x)
        return output
    
class Generator(nn.Module):
    def __init__(self,n_dims,max_resolution,lr,betas):
        out_channels = {
            4:512,
            8:512,
            16:256,
            32:512,
            64:266,
            128:128,
            256:64,
            512:32,
            1024:16
        }
        self.main = nn.ModuleList(BlockG(in_chanels=n_dims,out_channels=out_channels[4],is_input_layer=True))
        for index in range(2,int(log2(max_resolution))-1):
            self.main.append(BlockG(in_chanels=out_channels[index],out_channels=out_channels[index+1]))
        else:
            self.main.append(BlockG(in_chanels=out_channels[index+1],out_channels=3))

        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,betas=betas)

    def forward(self,x):
        output = self.main(x)
        return output





        


