import numpy 
import torch
from torch import nn
from gan_modules import *


class InputG(nn.Module):
    def __init__(self,in_channels,out_channels,negative_slope=0.1,is_spectral_norm=True):
        super().__init__()
        if(is_spectral_norm):
            Conv2d = sn_conv2d
            ConvT2d = sn_tconv2d
        else:
            Conv2d = EqualizedLRConv2d
            ConvT2d = EqualizedLRTConv2d
            
        self.main = nn.Sequential(
            ConvT2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4)),
            PixelNorm2d(),
            nn.LeakyReLU(negative_slope=negative_slope,inplace=True),
            Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            PixelNorm2d(),
            nn.LeakyReLU(negative_slope=negative_slope,inplace=True)
        )
    def forward(self,inputs):
        output = self.main(inputs)
        return output

class BlockG(nn.Module):
    def __init__(self,in_channels,out_channels,negative_slope=0.1,is_spectral_norm=True):
        super().__init__()
        if(is_spectral_norm):
            Conv2d = sn_conv2d
        else:
            Conv2d = EqualizedLRConv2d
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            PixelNorm2d(),
            nn.LeakyReLU(negative_slope=negative_slope,inplace=True),
            Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            PixelNorm2d(),
            nn.LeakyReLU(negative_slope=negative_slope,inplace=True),
        )
        
    def forward(self,x):
        output = self.main(x)
        return output

# need to get sample_size
class RGBAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0
        self.sample_size = 1
        self.const = 1
        
    def forward(self,RGBs):
        if(len(RGBs)==2):
            RGB,old_RGB = RGBs
        else:
            return RGBs[0]
        output = (1 - self.alpha)*old_RGB + self.alpha * RGB
        self.alpha += self.const
        if(self.alpha>1):
            self.alpha = 0
        return output
        
        
class Generator(nn.Module):
    def __init__(self,n_dims,max_resolution,negative_slope=0.1,is_spectral_norm=True):
        super().__init__()
        self.img_size = 4
        self.max_resolution = max_resolution
        self.is_spectral_norm = is_spectral_norm
        self.negative_slope = negative_slope
        self.out_channels={
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
        self.main = nn.ModuleList([
            InputG(
                in_channels=n_dims,
                out_channels=self.out_channels[4],
                negative_slope=negative_slope,
                is_spectral_norm=is_spectral_norm
                )
            ])
        if(is_spectral_norm):
            Conv2d = sn_conv2d
        else:
            Conv2d = EqualizedLRConv2d
        self.to_RGB = nn.ModuleDict({
            "up_to_date":Conv2d(in_channels=self.out_channels[4],out_channels=3,kernel_size=(1,1)),
            })
        self.output_layer =  nn.Sequential(RGBAdd(),nn.Tanh())

    def update(self):
        self.main.append(
            BlockG(
                in_channels=self.out_channels[self.img_size],
                out_channels=self.out_channels[self.img_size*2],
                negative_slope=self.negative_slope,
                is_spectral_norm=self.is_spectral_norm
                )
            )
        if(self.is_spectral_norm):
            Conv2d = sn_conv2d
            ConvT2d = sn_tconv2d
        else:
            Conv2d = EqualizedLRConv2d
            ConvT2d = EqualizedLRTConv2d
        self.to_RGB["old"] = nn.Sequential(
            ConvT2d(
                in_channels=self.out_channels[self.img_size],
                out_channels=self.out_channels[self.img_size],
                kernel_size=(2,2),
                stride=2),
            self.to_RGB["up_to_date"])
        self.to_RGB["up_to_date"] = Conv2d(in_channels=self.out_channels[self.img_size*2],out_channels=3,kernel_size=(1,1))
        self.img_size *= 2
        
    def forward(self,x):
        RGBs=[]
        for layer in self.main: 
            x = layer(x)
            if(self.img_size//2 == x.shape[-1]):
                RGBs.append(self.to_RGB["old"](x))
        RGBs.append(self.to_RGB["up_to_date"](x))
        output = self.output_layer(RGBs)
        return output
        
        
