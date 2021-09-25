import numpy as np
import torch
import warnings # to hide Named tensors warnings when uses max_pooling. This is a bug in pytorch https://github.com/pytorch/pytorch/issues/54846
from tqdm.notebook import tqdm 
from torch.nn import functional as F
from torch import nn
from gan_modules import *
from torch.utils import tensorboard
from math import log2

warnings.simplefilter("ignore")

def get_activation(activation,alpha=0.2):
    if(activation == "relu"):
        return nn.ReLU()
    elif(activation == "leakyrelu"):
        return nn.LeakyReLU(alpha)

class ResBlockG(nn.Module):
    def __init__(self,in_channels,out_channels,activation="leakyrelu",alpha=0.2,kernel_size=(3,3),upsampling_mode="tconv"):
        super().__init__()
        if(upsampling_mode=="tconv"):
            upsampling = sn_tconv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4),padding=1,stride=2,bias=False)
        else:
            upsampling = nn.Sequential(
                nn.Upsample(scale_factor=2),
                sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=1,bias=False)
            )
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            get_activation(activation,alpha),
            upsampling,
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
            sn_conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation,alpha),
            sn_conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=1,bias=False),
        )
        self.shutcut = nn.Sequential(
            upsampling,
            nn.BatchNorm2d(out_channels),
            get_activation(activation,alpha),
            sn_conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(1,1),bias=False),
        )

    def forward(self,inputs):
        main = self.main(inputs)
        shutcut = self.shutcut(inputs)
        output = main+shutcut
        return output


class Generator(nn.Module):
    def __init__(self,n_dims=512,max_resolutions=256,lr=0.01,betas=(0.999),initial_layer="tconv",upsampling_mode="tconv",attention_loc=32):
        super().__init__()
        self.n_dims = n_dims
        self.initial_layer = initial_layer
        if(initial_layer=="tconv"):
            self.inputs = sn_tconv2d(in_channels=n_dims,out_channels=n_dims,kernel_size=(4,4),bias=False)
        else:
            self.inputs = sn_linear(in_features=n_dims,out_features=n_dims*4*4,bias=False)
        out_channels = {
            4:512,
            8:256,
            16:256,
            32:256,
            64:128,
            128:32,
            256:16,
            512:8
        }
        self.layers = nn.ModuleList([ResBlockG(in_channels=self.n_dims, out_channels=out_channels[4],upsampling_mode=upsampling_mode)]) 
        for index in range(3,int(log2(max_resolutions))-1):
            self.layers.append(ResBlockG(in_channels=out_channels[2**(index-1)],out_channels=out_channels[2**(index)],upsampling_mode=upsampling_mode))
            if(attention_loc == 2**(index)):
                self.layers.append(SelfAttention(in_channels=out_channels[2**(index)]))
        else:
            self.layers.append(ResBlockG(in_channels=out_channels[2**(index)], out_channels=3,upsampling_mode=upsampling_mode))
            self.layers.append(nn.Tanh())
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,betas=betas)
        
    def forward(self,inputs):
        x = self.inputs(inputs)
        if(self.initial_layer != "tconv"):
            x = x.view(-1,self.n_dims,4,4)
        for layer in self.layers:
            x = layer(x)
        return x

        
class ResBlockD(nn.Module):
    def __init__(self,in_channels,out_channels,activation="leakyrelu",alpha=0.2,downsampling_mode="conv"):
        super().__init__()
        self.downsampling_mode = downsampling_mode
        self.main = nn.Sequential(
            get_activation(activation,alpha),
            sn_conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(3,3),padding=1),
            get_activation(activation,alpha),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
        )
        self.shutcut = nn.Sequential(
            get_activation(activation,alpha),
            sn_conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1))
        )

        if(downsampling_mode == "conv"):
            self.downsampling_main = sn_conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(4,4),padding=1,stride=2)
            self.downsampling_shutcut = sn_conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(4,4),padding=1,stride=2)
    
    def forward(self,inputs):
        main = self.main(inputs)
        shutcut = self.shutcut(inputs)
        if(self.downsampling_mode=="conv"):
            output = self.downsampling_main(main)+self.downsampling_shutcut(shutcut)
        elif(self.downsampling_mode=="pooling"):
            _,_,h,w = inputs.shape
            output = F.adaptive_avg_pool2d(main, output_size=(h//2,w//2)) + F.adaptive_avg_pool2d(shutcut, output_size=(h//2,w//2))
        else:
            output = main + shutcut
        return output

class Discriminator(nn.Module):
    def __init__(self,max_resolutions=256,lr=0.01,betas=(0,0.999),activation="leakyrelu",downsampling_mode="conv",attention_loc=32):
        super().__init__()
        out_channels={
            4:1024,
            8:1024,
            16:512,
            32:256,
            64:128,
            128:128,
            256:16,
            512:8,
        }
        self.layers = nn.ModuleList([ResBlockD(in_channels=3, out_channels=out_channels[max_resolutions],activation=activation,downsampling_mode=downsampling_mode)])
        for index in range(int(log2(max_resolutions))-4):
            self.layers.append(ResBlockD(in_channels=out_channels[max_resolutions//2**(index)], out_channels=out_channels[max_resolutions//2**(index+1)],activation=activation,downsampling_mode=downsampling_mode))
            if((max_resolutions//2**(index+1))==attention_loc):
                self.layers.append(SelfAttention(in_channels=out_channels[max_resolutions//2**(index+1)]))
        else:
            self.layers.extend([
                ResBlockD(in_channels=out_channels[max_resolutions//2**(index+1)], out_channels=out_channels[max_resolutions//2**(index+2)],activation=activation,downsampling_mode=downsampling_mode),
                MiniBatchStddev(),
                ResBlockD(in_channels=out_channels[max_resolutions//2**(index+2)]+1, out_channels=out_channels[max_resolutions//2**(index+2)],activation=activation,downsampling_mode=None),
                get_activation(activation),
                GlobalSum(),
                sn_linear(in_features=out_channels[max_resolutions//2**(index+2)],out_features=1)
            ])
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr,betas=betas)
        
    def forward(self,inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class SAGAN(GAN):
    def __init__(
            self,
            n_dims,
            n_dis,
            max_resolutions,
            g_lr=1e-3,
            d_lr=1e-3,
            g_betas=(0,0.9),
            d_betas=(0,0.9),
            initial_layer="tconv",
            upsampling_mode="tconv",
            downsampling_mode="conv",
            loss  = "hinge",
            attention_loc=32
        ):
        super().__init__()
        self.n_dis = n_dis
        self.initial_layer = initial_layer
        self.device = "cuda" if (torch.cuda.is_available()) else "cpu"
        self.netD = Discriminator(max_resolutions,d_lr,d_betas,downsampling_mode=downsampling_mode,attention_loc=attention_loc).to(self.device)
        self.netG = Generator(n_dims,max_resolutions,g_lr,g_betas,initial_layer=initial_layer,upsampling_mode=upsampling_mode,attention_loc=attention_loc).to(self.device)
        self.n_dims = n_dims
        self.loss = Hinge() if loss == "hinge" else WassersteinGP(self.netD)
        if(self.initial_layer=="tconv"):
            self.fixed_noise = torch.randn(size=(16,self.n_dims,1,1),device=self.device)
        else:
            self.fixed_noise = torch.randn(size=(16,self.n_dims),device=self.device)

    def train_d(self,real_img,fake_img):
        self.netD.optimizer.zero_grad()
        real = self.netD(real_img)
        fake = self.netD(fake_img)
        loss = self.loss(real,fake,real_img,fake_img)
        loss.backward()
        self.netD.optimizer.step()
        return loss
        
    def train_g(self,fake_img):
        self.netG.optimizer.zero_grad()
        fake = self.netD(fake_img)
        loss = -fake.mean()
        loss.backward()
        self.netG.optimizer.step()
        return loss
    
    def generate(self):
        if(self.initial_layer=="tconv"):
            noise = torch.randn(size=(1,self.n_dims,1,1),device=self.device)
        else:
            noise = torch.randn(size=(1,self.n_dims),device=self.device)
        with torch.no_grad():
            img = (torch.squeeze(self.netG(noise),dim=0) +1)/2
        return img

    def fit(self,dataset,epochs,batch_size=10,shuffle=True,num_workers=0,is_tensorboard=True):
        if(is_tensorboard):
            log_writer = tensorboard.SummaryWriter(log_dir="./logs")
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        save_num = len(loader) // 10
        for epoch in tqdm(range(epochs),desc="Epochs",total=epochs+self.total_epochs,initial=self.total_epochs):
            for step,data in enumerate(tqdm(loader,desc="Steps",leave=False),start=1):
                real_imgs,_ = data
                real_imgs = real_imgs.to(self.device)
                if(self.initial_layer=="tconv"):
                    noise = torch.randn(size=(real_imgs.shape[0],self.n_dims,1,1),device=self.device)
                else:
                    noise = torch.randn(size=(real_imgs.shape[0],self.n_dims),device=self.device)
                with torch.no_grad():
                    fake_imgs = self.netG(noise)
                loss_d = self.train_d(real_imgs, fake_imgs).to("cpu")
                if(self.initial_layer=="tconv"):
                    noise = torch.randn(size=(real_imgs.shape[0],self.n_dims,1,1),device=self.device)
                else:
                    noise = torch.randn(size=(real_imgs.shape[0],self.n_dims),device=self.device)
                fake_imgs = self.netG(noise)
                self.total_steps += 1
                if(self.total_steps%self.n_dis==0):
                    loss_g = self.train_g(fake_imgs).to("cpu")
                else:
                    with torch.no_grad():
                        loss_g = -self.netD(fake_imgs).mean().to("cpu")
                if(is_tensorboard):
                    log_writer.add_scalars(
                        "Loss",
                        {"Generator":loss_g.item(),"Discriminator":loss_d.item()},
                        global_step=self.total_steps
                        )

                if(step % save_num == 0):
                    with torch.no_grad():
                        sample_img = (self.netG(self.fixed_noise).detach() + 1) / 2
                        sample_img = sample_img.to("cpu")
                    save_img(sample_img, file_name=f"epoch{epoch+1+self.total_epochs}_step{step}")
                    for i in range(5):
                        if(self.initial_layer=="tconv"):
                            check_vec = torch.randn(size=(1,self.n_dims,1,1),device=self.device)
                        else:
                            check_vec = torch.randn(size=(1,self.n_dims),device=self.device)
                        with torch.no_grad():
                            check_img = (torch.squeeze(self.netG(check_vec)).detach() + 1) / 2
                            check_img = check_img.to("cpu")
                        save_img(check_img, file_name=f"epoch{epoch+1+self.total_epochs}_step{step}_check{i}",is_grid=False)
                    else:
                        del check_vec
            self.save_model(f"params\epoch_{epoch+self.total_epochs}")

        if(is_tensorboard):
            log_writer.close()
