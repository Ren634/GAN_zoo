from ipywidgets.widgets.widget_string import Label
import numpy as np
import torch
from gan_modules import * 
from torch import nn
from torch.nn import functional as F
from math import log2
from collections import deque
import random
import lpips
import warnings
from tqdm._tqdm_notebook import tqdm
from torch.utils import tensorboard
device = "cuda" if torch.cuda.is_available else "cpu"
warnings.simplefilter("ignore")

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
            nn.BatchNorm2d(out_channels*2),
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
            4:1024,
            8:512,
            16:256,
            32:256,
            64:128,
            128:64,
            256:32,
            512:16,
            1024:8
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
        self.output_layer_128 = nn.Sequential(
            sn_conv2d(in_channels=out_channels[128],out_channels=3,kernel_size=(3,3),stride=1,padding=1),
            nn.Tanh()
        )
        self.output_layer = nn.Sequential(
            sn_conv2d(in_channels=out_channels[max_resolution],out_channels=3,kernel_size=(3,3),stride=1,padding=1),
            nn.Tanh()
        )
        self.intermidiate = deque()
    def forward(self,x,output_mid=False):
        for layer in self.feat128:
            x = layer(x)
            if(8<=x.shape[-1]<=2**(len(self.SLE_layer) + 2)):
                self.intermidiate.append(x)
        if(output_mid):
            x_128 = self.output_layer_128(x)
        for layer in self.feat128_to_max:
            if(len(self.intermidiate)>0):
                x = self.SLE_layer[len(self.intermidiate)-1](self.intermidiate.popleft(),x)
            x = layer(x)
        else:
            if(len(self.intermidiate)>0):
                x = self.SLE_layer[len(self.intermidiate)-1](self.intermidiate.popleft(),x)
        output = self.output_layer(x)
        return (output,x_128) if (output_mid) else output

class Discriminator(nn.Module):
    def __init__(self,max_resolution,recon_size=128,leakyrelu_alpha=0.1):
        super().__init__()
        out_channels= {
            4:1024,
            8:512,
            16:512,
            32:256,
            64:128,
            128:128,
            256:64,
            512:32,
            1024:16
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
        self.max_resolution = max_resolution
        self.SLE_layer = nn.ModuleList([
            SLE(in_channels=3,out_channels=out_channels[max_resolution//4]),
            SLE(in_channels=out_channels[max_resolution//2],out_channels=out_channels[max_resolution//8])
            ])
        for index in range(int(log2(max_resolution//4))-5):
            self.feat8.insert(2+index,BlockD(in_channels=out_channels[max_resolution//2**(index+2)],out_channels=out_channels[max_resolution//2**(index+3)],leakyrelu_alpha=leakyrelu_alpha))
        self.feat8.extend([
            BlockD(in_channels=out_channels[32],out_channels=out_channels[16],leakyrelu_alpha=leakyrelu_alpha),
            BlockD(in_channels=out_channels[16],out_channels=out_channels[8],leakyrelu_alpha=leakyrelu_alpha),
        ])
        def output_block():
            return nn.Sequential(
            sn_conv2d(in_channels=out_channels[8],out_channels=out_channels[4],kernel_size=(1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels[4]),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True),
            sn_conv2d(in_channels=out_channels[4],out_channels=out_channels[4],kernel_size=(1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels[4]),
            nn.LeakyReLU(leakyrelu_alpha,inplace=True),
            sn_conv2d(in_channels=out_channels[4],out_channels=1,kernel_size=(4,4),stride=1,padding=0)
        )

        self.output_layer = output_block()       
        self.feat8_128 = nn.ModuleList([
            BlockD(in_channels=3,out_channels=out_channels[64]),
            BlockD(in_channels=out_channels[64],out_channels=out_channels[32]),
            BlockD(in_channels=out_channels[32],out_channels=out_channels[16]),
            BlockD(in_channels=out_channels[16],out_channels=out_channels[8]), 
            output_block()
        ])
        self.simple_decoderx16 = SimpleDecoder(in_channels=out_channels[16])
        self.simple_decoderx8 = SimpleDecoder(in_channels=out_channels[8]) 
        self.simple_decoder_mid = SimpleDecoder(in_channels=out_channels[8])
        self.intermidiate = deque()

    def forward(self,x,label=None):
        if(isinstance(x,(list,tuple))):
            x,x_mid = x
        else:
            x_mid = F.interpolate(x,size=self.recon_size)
        for layer in self.feat8_128:
            x_mid = layer(x_mid)
            if(label == "real" and x_mid.shape[-1] == 8):
                recon_mid = [None,self.simple_decoder_mid(x_mid)]
        self.intermidiate.append(x)
        if(label == "real"):
            crop_loc = random.randint(0,3)
            recon_16 = [F.interpolate(crop(x,crop_loc),size=self.recon_size),None]
            recon_8 = [F.interpolate(x,size=self.recon_size),None]
            recon_mid[0] = F.interpolate(x_mid,size=self.recon_size)
        for layer in self.feat8:
            x = layer(x) 
            if(x.shape[-1] == self.max_resolution//2):
                self.intermidiate.append(x)
            if(x.shape[-1] == self.max_resolution//4 or x.shape[-1] == self.max_resolution//8):
                x = self.SLE_layer[-len(self.intermidiate)](self.intermidiate.popleft(),x)
            if(label == "real" and x.shape[-1] == 16):
                x16 = crop(x,crop_loc)
        if(label == "real"):
            recon_16[1] = self.simple_decoderx16(x16) 
            recon_8[1] = self.simple_decoderx8(x)
        output = self.output_layer(x)
        return (torch.cat([output,x_mid]),recon_16,recon_8,recon_mid) if label == "real" else torch.cat([output,x_mid])

class LightWeightGAN(GAN):
    def __init__(
        self,
        n_dims=256,
        n_dis=1,
        max_resolution=512,
        g_lr=0.01,
        d_lr=0.01,
        g_betas=(0,0.999),
        d_betas=(0,0.999),
        is_da = True,
        recon_loss = "vgg",
        recon_size=128
        ):
        super().__init__()
        self.n_dims = n_dims
        self.n_dis = n_dis
        self.netD = Discriminator(max_resolution,recon_size=recon_size).to(device)
        self.netG = Generator(n_dims,max_resolution).to(device)
        self.optimizer_d = torch.optim.Adam(self.netD.parameters(),lr=d_lr,betas=d_betas)
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(),lr=g_lr,betas=g_betas)
        self.adversarial_loss = hinge
        self.recon_loss = lpips.LPIPS(net="vgg").to(device) if recon_loss == "vgg" else hinge
        self.is_da = is_da
        if(is_da):
            self.data_aug = AdaptiveDA(self.netD,const=0.01)
        
    def train_d(self,real_img,fake_img):
        self.optimizer_d.zero_grad()
        if(self.is_da):
            real_img = self.data_aug.apply(real_img,target=True)
        fake_output = self.netD(fake_img)
        real_output,recon_16,recon_8,recon_mid = self.netD(real_img,label="real")
        loss = hinge(real_output,fake_output)
        if(self.is_da):
            loss += self.recon_loss(*recon_16) + self.recon_loss(*recon_8) + self.recon_loss(*recon_mid)
        loss.backward()
        self.optimizer_d.step()
        return loss
        
    def train_g(self,fake_img):
        self.optimizer_g.zero_grad()
        fake_output = self.netD(fake_img)
        loss = -torch.mean(fake_output)
        loss.backward()
        self.optimizer_g.step()
        return loss

    def fit(self,dataset,epochs,batch_size=10,shuffle=True,num_workers=0,is_tensorboard=True,image_num=10):
        if(is_tensorboard):
            log_writer = tensorboard.SummaryWriter(log_dir="./logs")
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        save_num = len(loader) // image_num
        self.fixed_noise = torch.randn(size=(16,self.n_dims,1,1),device=device)
        for epoch in tqdm(range(epochs),desc="Epochs",total=epochs+self.total_epochs,initial=self.total_epochs):
            for step,data in enumerate(tqdm(loader,desc="Steps",leave=False),start=1):
                real_imgs,_ = data
                real_imgs = real_imgs.to(device)
                noise = torch.randn(size=(real_imgs.shape[0],self.n_dims,1,1),device=device)
                fake_imgs = self.netG(noise,True)
                if(self.is_da):
                    fake_imgs = [self.data_aug.apply(fake_img) for fake_img in fake_imgs]
                loss_d = self.train_d(real_imgs,[fake_img.detach() for fake_img in fake_imgs]).to("cpu")
                self.total_steps += 1
                if(self.total_steps > 1e+5):
                    self.is_da = False
                if(self.total_steps % self.n_dis ==0):
                    loss_g = self.train_g(fake_imgs).to("cpu")
                else:
                    with torch.no_grad():
                        output = self.netD(fake_imgs)
                    loss_g = -torch.mean(output).to("cpu")
                    del output
                if(is_tensorboard):
                    log_writer.add_scalars(
                        "Loss",
                        {"Generator":loss_g.item(),"Discriminator":loss_d.item()},
                        global_step=self.total_steps
                        )
                    log_writer.add_scalars(
                        "ADA",
                        {"p":self.data_aug._apply_p,"rt":self.data_aug.rt},
                        global_step=self.total_steps
                        )
                    
                if(step % save_num == 0):
                    with torch.no_grad():
                        sample_img = (self.netG(self.fixed_noise).detach() + 1) / 2
                        sample_img = sample_img.to("cpu")
                    save_img(sample_img, file_name=f"epoch{self.total_epochs}_step{step}")
                    for i in range(5):
                        check_vec = torch.randn(size=(1,self.n_dims,1,1),device=device)
                        with torch.no_grad():
                            check_img = (torch.squeeze(self.netG(check_vec)).detach() + 1) / 2
                            check_img = check_img.to("cpu")
                        save_img(check_img, file_name=f"epoch{self.total_epochs}_step{step}_check{i}",is_grid=False)
                    else:
                        del check_vec
            self.total_epochs += 1
            if((epoch+1) % 10 == 0):
                self.save_model(f"params\epoch_{self.total_epochs-1}")

        if(is_tensorboard):
            log_writer.close()

            



        
        
        
