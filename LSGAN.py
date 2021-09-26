import torch
from torch.nn import functional as F
from torch import nn
from gan_modules import *
from math import log2

class BlockG(nn.Module):
    def __init__(self,in_chanels,out_channels,is_input_layer=False,activation="leakyrelu"):
        super().__init__()
        if(is_input_layer):
            ConvT2d = nn.ConvTranspose2d(in_channels=in_chanels,out_channels=out_channels,kernel_size=(4,4),bias=False)
        else:
            ConvT2d = nn.ConvTranspose2d(in_channels=in_chanels,out_channels=out_channels,kernel_size=(2,2),stride=2,bias=False)
        if(activation=="tanh"):
            act_func = nn.Tanh()
        else:
            act_func = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.main = nn.Sequential(
            ConvT2d,
            nn.BatchNorm2d(out_channels),
            act_func
        )

    def forward(self,x):
        output = self.main(x)
        return output
    
class Generator(nn.Module):
    def __init__(self,n_dims,max_resolution,lr,betas):
        super().__init__()
        out_channels = {
            4:512,
            8:512,
            16:512,
            32:512,
            64:256,
            128:128,
            256:64,
            512:32,
            1024:16
        }
        self.main = nn.ModuleList([BlockG(in_chanels=n_dims,out_channels=out_channels[4],is_input_layer=True)])
        end = int(log2(max_resolution)) -1
        for index in range(2,end):
            self.main.append(BlockG(in_chanels=out_channels[2**index],out_channels=out_channels[2**(index+1)]))
        else:
            self.main.append(BlockG(in_chanels=out_channels[2**(index+1)],out_channels=3,activation="tanh"))

        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,betas=betas)

    def forward(self,x):
        for layer in self.main:
            x = layer(x)
        return x

class BlockD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4),stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self,x):
        output = self.main(x)
        return output

class Discriminator(nn.Module):
    def __init__(self,max_resolution,lr,betas):
        super().__init__()
        out_channels = {
            4:512,
            8:512,
            16:512,
            32:512,
            64:256,
            128:128,
            256:64,
            512:32,
            1024:16
        }
        self.main = nn.ModuleList([BlockD(in_channels=3,out_channels=out_channels[max_resolution//2])])
        end = int(log2(max_resolution))-2
        for index in range(1,end):
            self.main.append(
                BlockD(in_channels=out_channels[max_resolution//2**(index)],out_channels=out_channels[max_resolution//2**(index+1)])
                )
        else:
            self.main.extend([
                nn.Flatten(),
                nn.Linear(in_features=out_channels[max_resolution//2**(index+1)]*4*4,out_features=1)
            ])
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,betas=betas)

    def forward(self,x):
        for layer in self.main:
            x = layer(x)
        return x
            
    




        


