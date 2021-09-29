import torch
from torch import nn
from gan_modules import *
from math import log2
from torch.utils import tensorboard
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"

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
            
class LSGAN(GAN):
    def __init__(
        self,
        n_dims,
        n_dis,
        max_resolution,
        g_lr,
        d_lr,
        g_betas,
        d_betas,
        is_moving_average,
        is_DA,
        ):
        super().__init__()
        self.netD = Discriminator(max_resolution,d_lr,d_betas).to(device)
        self.netG = Generator(n_dims,max_resolution,g_lr,g_betas).to(device)
        self.fixed_noise = torch.randn(size=(16,n_dims,1,1),device=device)
        self.n_dims = n_dims
        self.n_dis = n_dis
        self.is_moving_average = is_moving_average
        self.is_DA = is_DA
        if(is_DA):
            self.aug = AdaptiveDA(device=device)
        if(is_moving_average):
            self.ema = EMA()
            self.mvag_netG = Generator(n_dims,max_resolution,g_lr,g_betas).to(device)
            self.ema.setup(self.mvag_netG)
            
    def train_d(self,real_img,fake_img):
        self.netD.optimizer.zero_grad()
        if(self.is_DA):
            real_img = self.aug(real_img,target=True)
            fake_img = self.aug(fake_img)
        real = self.netD(real_img)
        fake = self.netD(fake_img)
        loss = 0.5*torch.square(real-1).mean() + 0.5*torch.square(fake).mean()
        loss.backward()
        self.netD.optimizer.step()
        if(self.is_DA):
            self.aug.adjust_p(real)
        return loss
    
    def train_g(self,fake_img):
        self.netG.optimizer.zero_grad()
        if(self.is_DA):
            fake_img = self.aug(fake_img)
        fake = self.netD(fake_img)
        loss = 0.5 * torch.square(fake-1).mean()
        loss.backward()
        self.netG.optimizer.step()
        if(self.is_moving_average):
            self.ema.apply(self.netG,self.mvag_netG)
        return loss

    def generate(self,img_num=1):
        netG = self.mvag_netG if self.is_moving_average else self.netG
        with torch.no_grad():
            x = torch.randn(size=(img_num,self.n_dims,1,1))
            output = netG(x)
        return netG
    
    def fit(self,dataset,epochs,batch_size=10,shuffle=True,num_workers=0,is_tensorboard=True,image_num=100):
        if(is_tensorboard):
            log_writer = tensorboard.SummaryWriter(log_dir="./logs")
        if(self.is_DA):
            self.aug.epoch = epochs
            self.aug.batch_size = batch_size
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        save_num = len(loader) // image_num
        for _ in tqdm(range(epochs),desc="Epochs",total=epochs+self.total_epochs,initial=self.total_epochs):
            for step,data in enumerate(tqdm(loader,desc="Steps",leave=False),start=1):
                real_imgs,_ = data
                real_imgs = real_imgs.to(device)
                noise = torch.randn(size=(real_imgs.shape[0],self.n_dims,1,1),device=device)
                with torch.no_grad():
                    fake_imgs = self.netG(noise)
                loss_d = self.train_d(real_imgs, fake_imgs).to("cpu")
                noise = torch.randn(size=(real_imgs.shape[0],self.n_dims,1,1),device=device)
                fake_imgs = self.netG(noise)
                self.total_steps += 1
                if(self.total_steps%self.n_dis==0):
                    loss_g = self.train_g(fake_imgs).to("cpu")
                else:
                    with torch.no_grad():
                        loss_g = -0.5*torch.square(self.netD(fake_imgs)-1).mean()
                if(is_tensorboard):
                    log_writer.add_scalars(
                        "Loss",
                        {"Generator":loss_g.item(),"Discriminator":loss_d.item()},
                        global_step=self.total_steps
                        )

                if(step % save_num == 0):
                    netG = self.mvag_netG if self.is_moving_average else self.netG
                    with torch.no_grad():
                        sample_img = (netG(self.fixed_noise).detach() + 1) / 2
                        sample_img = sample_img.to("cpu")
                    save_img(sample_img, file_name=f"epoch{self.total_epochs}_step{step}")
                    for i in range(5):
                        check_vec = torch.randn(size=(1,self.n_dims,1,1),device=device)
                        with torch.no_grad():
                            check_img = (torch.squeeze(netG(check_vec)).detach() + 1) / 2
                            check_img = check_img.to("cpu")
                        save_img(check_img, file_name=f"epoch{self.total_epochs}_step{step}_check{i}",is_grid=False)
                    else:
                        del check_vec
            if(self.total_epochs % 500 == 0):
                self.save_model(f"params\epoch_{self.total_epochs}")
            self.total_epochs +=1

        if(is_tensorboard):
            log_writer.close()







        


