import numpy as np
import torch
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from gan_modules import * 
from torch import nn
from torch.nn import functional as F
from math import log2
from collections import deque
import random
import lpips
from tqdm._tqdm_notebook import tqdm
from torch.utils import tensorboard

def crop(x,crop_loc):
    _,_,h,w = x.shape
    h_half,w_half = h//2,w//2
    if(crop_loc == 0):
        output = x[:,:,:h_half,:w_half]
    elif(crop_loc == 1):
        output = x[:,:,:h_half,w_half:]
    elif(crop_loc == 2):
        output = x[:,:,h_half:,:w_half]
    elif(crop_loc == 3):
        output = x[:,:,h_half:,w_half:]
    return output

class SLE(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            sn_conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(4,4),stride=1,padding=0),
            nn.LeakyReLU(0.1,inplace=True),
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
    
class BlockD(nn.Module):
    def __init__(self,in_channels,out_channels,leakyrelu_alpha=0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.AvgPool2d((2,2)),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True)
        )
        self.branch = nn.Sequential(
            sn_conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(4,4),stride=2,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True),
        )
    
    def forward(self,x):
        main = self.main(x)
        branch = self.branch(x)
        return main + branch

class SimpleDecoder(nn.Module):
    def __init__(self,in_channels): 
        super().__init__()
        def block(in_channels,out_channels):
            block = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels*2,kernel_size=(3,3),stride=1,padding=1,bias=False),
            BatchNorm2d(out_channels*2),
            nn.GLU(dim=1)
            )
            return block
        out_channels={
            8:256,
            16:128,
            32:64,
            64:32,
        }
        self.main = nn.Sequential(
            block(in_channels,out_channels[8]),
            block(out_channels[8],out_channels[16]),
            block(out_channels[16],out_channels[32]),
            block(out_channels[32],out_channels[64]),
            sn_conv2d(in_channels=32,out_channels=3,kernel_size=(3,3),stride=1,padding=1),
            nn.Tanh() 
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
        self.output_layer_128 = nn.Sequential(
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

class Discriminator(nn.Module):
    def __init__(self,max_resolution,recon_size=128,leakyrelu_alpha=0.1):
        super().__init__()
        out_channels= {
            4:1024,
            8:512,
            16:512,
            32:256,
            64:128,
            128:64,
            256:32,
            512:3,
            1024:3
        }
        self.recon_size = recon_size
        self.feat8 = nn.ModuleList([
            nn.Sequential(
            sn_conv2d(in_channels=3,out_channels=out_channels[max_resolution//2],kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True)
            ),
            nn.Sequential(
            sn_conv2d(in_channels=out_channels[max_resolution//2],out_channels=out_channels[max_resolution//4],kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(out_channels[max_resolution//4]),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True)
            )
        ])
        for index in range(int(log2(max_resolution//4))-5):
            self.feat8.insert(2+index,BlockD(in_channels=out_channels[max_resolution//2**(index+2)],out_channels=out_channels[max_resolution//2**(index+3)],leakyrelu_alpha=leakyrelu_alpha))
        self.feat8.extend([
            BlockD(in_channels=out_channels[32],out_channels=out_channels[16],leakyrelu_alpha=leakyrelu_alpha),
            BlockD(in_channels=out_channels[16],out_channels=out_channels[8],leakyrelu_alpha=leakyrelu_alpha),
        ])
        self.output_layer = nn.Sequential(
            sn_conv2d(in_channels=out_channels[8],out_channels=out_channels[4],kernel_size=(1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels[4]),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True),
            sn_conv2d(in_channels=out_channels[4],out_channels=1,kernel_size=(4,4),stride=1,padding=0)
        ) 
        self.simple_decoderx16 = SimpleDecoder(in_channels=out_channels[16])
        self.simple_decoderx8 = SimpleDecoder(in_channels=out_channels[8]) 

    def forward(self,x,label="fake"):
        for layer in self.feat8:
            x = layer(x) 
            if(label == "fake" and x.shape[-1] == 16):
                self.crop_loc = random.randint(0,3)
                x16 = crop(x,self.crop_loc)
        if(label == "fake"):
            recon_16 = self.simple_decoderx16(x16)
            recon_8 = self.simple_decoderx8(x)
        elif(label == "real"):
            recon_8 = F.interpolate(x,(self.recon_size,self.recon_size))
            recon_16 = F.interpolate(crop(x,self.crop_loc),(self.recon_size,self.recon_size)) 
        else:
            recon_16,recon_8 = None,None
        output = self.output_layer(x)
        return output,(recon_16,recon_8)

class LightWeightGAN(GAN):
    def __init__(
        self,
        n_dims,
        n_dis,
        max_resolution,
        g_lr,
        d_lr,
        g_betas,
        d_betas,
        is_da,
        recon_size=128):
        super().__init__()
        self.netD = Discriminator(max_resolution,recon_size=recon_size)
        self.netG = Generator(n_dims,max_resolution)
        self.optimizer_d = torch.optim.Adam(self.netD.parameters(),lr=d_lr,betas=d_betas)
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(),lr=g_lr,betas=g_betas)
        self.adversarial_loss = hinge
        self.recon_loss = lpips.LPIPS()
        
    def train_d(self,real_img,fake_img):
        self.optimizer_d.zero_grad()
        fake_output,fake_recon = self.netD(fake_img,label="fake")
        real_output,real_recon = self.netD(real_img,label="real")
        loss = hinge(real_output,fake_output)
        loss += self.recon_loss(torch.cat(real_recon),torch.cat(fake_recon))
        loss.backward()
        self.optimizer_d.step()
        
    def train_g(self,fake_img):
        self.optimizer_g.zero_grad()
        fake_output,_ = self.netD(fake_img,label=None)
        loss = torch.mean(fake_output)
        loss.backward()
        self.optimizer_g.step()



            



        
        
        
