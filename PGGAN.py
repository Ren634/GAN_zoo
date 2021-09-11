import torch
from torch import nn
from gan_modules import *
from tqdm.notebook import tqdm
from torchvision.transforms import functional as TF
from torch.utils import tensorboard
from math import log2

device = "cuda" if torch.cuda.is_available else "cpu"

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
            nn.LeakyReLU(negative_slope=negative_slope),
            Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            PixelNorm2d(),
            nn.LeakyReLU(negative_slope=negative_slope)
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
            nn.LeakyReLU(negative_slope=negative_slope),
            Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            PixelNorm2d(),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        
    def forward(self,x):
        output = self.main(x)
        return output

class RGBAdd(nn.Module):
    def __init__(self,sample_size):
        super().__init__()
        self.alpha = 0
        self.const = 1 / sample_size
        
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
        self.img_size = torch.tensor(4)
        self.register_buffer("image_size",self.img_size)
        self.max_resolution = max_resolution
        self.is_spectral_norm = is_spectral_norm
        self.negative_slope = negative_slope
        self.__sample_size = 1
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
        
        self.output_layer =  nn.Sequential(RGBAdd(self.sample_size),nn.Tanh())

    @property
    def sample_size(self):
        return self.__sample_size
    
    @sample_size.setter
    def sample_size(self,value):
        self.__sample_size = value
        self.output_layer = nn.Sequential(RGBAdd(value),nn.Tanh())
        
    def update(self):
        img_size = self.img_size.item()
        self.main.append(    
            BlockG(
                in_channels=self.out_channels[img_size],
                out_channels=self.out_channels[img_size*2],
                negative_slope=self.negative_slope,
                is_spectral_norm=self.is_spectral_norm
                ).to(device)
            )
        if(self.is_spectral_norm):
            Conv2d = sn_conv2d
        else:
            Conv2d = EqualizedLRConv2d
             
        self.to_RGB["old"] = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            self.to_RGB["up_to_date"]).to(device)
        self.to_RGB["up_to_date"] = Conv2d(in_channels=self.out_channels[img_size*2],out_channels=3,kernel_size=(1,1)).to(device)
        self.img_size *= 2
        
    def forward(self,x):
        RGBs= []
        for layer in self.main: 
            x = layer(x)
            if(self.img_size.item()//2 == x.shape[-1]):
                RGBs.append(self.to_RGB["old"](x))
        RGBs.append(self.to_RGB["up_to_date"](x))
        output = self.output_layer(RGBs)
        return output
    
class BlockD(nn.Module):
    def __init__(self,in_channels,out_channels,negative_slope=0.1,is_spectral_norm=True):
        super().__init__()
        if(is_spectral_norm):
            Conv2d = sn_conv2d
        else:
            Conv2d = EqualizedLRConv2d
        self.main = nn.Sequential(
            Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(negative_slope=negative_slope),
            Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.AvgPool2d(kernel_size=(2,2))
        )
    def forward(self,inputs):
        output = self.main(inputs)
        return output

class Discriminator(nn.Module):
    def __init__(self,negative_slope=0.1,is_spectral_norm=True):
        super().__init__()
        if(is_spectral_norm):
            Conv2d = sn_conv2d
            Linear = sn_linear
        else:
            Conv2d = EqualizedLRConv2d 
            Linear = EqualizedLRLinear
        self.is_spectral_norm = is_spectral_norm
        self.img_size = torch.tensor(4)
        self.register_buffer("image_size",self.img_size)
        self.negative_slope = negative_slope
        self.__sample_size = 1
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
        self.fromRGB = nn.ModuleDict({
            "up_to_date":nn.Sequential(
            Conv2d(in_channels=3,out_channels=self.out_channels[4],kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=negative_slope)
            )
        })
        self.main = nn.ModuleList()
        self.output_layer = nn.Sequential(
            MiniBatchStddev(),
            Conv2d(in_channels=self.out_channels[4]+1,out_channels=self.out_channels[4],kernel_size=(3,3),padding=1),
            nn.LeakyReLU(negative_slope=negative_slope),
            Conv2d(in_channels=self.out_channels[4],out_channels=self.out_channels[4],kernel_size=(4,4)),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Flatten(),
            Linear(in_features=self.out_channels[4],out_features=1)
            )
        self.add_fromRGB = RGBAdd(self.__sample_size)
    
    @property
    def sample_size(self):
        return self.__sample_size
    
    @sample_size.setter
    def sample_size(self,value):
        self.__sample_size = value
        self.add_fromRGB = RGBAdd(value)        

    def update(self):
        img_size = self.img_size.item()
        if(self.is_spectral_norm):
            Conv2d = sn_conv2d
        else:
            Conv2d = EqualizedLRConv2d 
            
        self.fromRGB["old"] = nn.Sequential(
            nn.AvgPool2d(2,2),
            self.fromRGB["up_to_date"]
        ).to(device)
        
        self.fromRGB["up_to_date"] = nn.Sequential(
            Conv2d(
                in_channels=3,
                out_channels=self.out_channels[img_size*2],
                kernel_size=(1,1)
                ),
            nn.LeakyReLU(negative_slope=self.negative_slope)
        ).to(device)
        self.main.insert(0,BlockD(in_channels=self.out_channels[img_size*2],out_channels=self.out_channels[img_size]).to(device))
        self.img_size *= 2
         
    def forward(self,x):
        fromRGBs = []
        up_to_date_RGB= self.fromRGB["up_to_date"](x)
        for layer in self.main:
            up_to_date_RGB = layer(up_to_date_RGB)
            if(up_to_date_RGB.shape[-1]==self.img_size):
                fromRGBs.append(up_to_date_RGB)
                if(len(self.fromRGB)>1):
                    fromRGBs.append(self.fromRGB["old"](x))
                    up_to_date_RGB = self.add_fromRGB(fromRGBs)
        output = self.output_layer(up_to_date_RGB) 
        return output

class PGGAN(GAN):
    def __init__(
        self,
        n_dims,
        max_resolution,
        g_lr=0.01,
        d_lr=0.01,
        d_betas=(0,0.999),
        g_betas=(0,0.999),
        negative_slope=0.1,
        is_spectral_norm=True,
        is_moving_average=True,
        loss ="wasserstein",
        ):
        super().__init__()            
        self.max_resolution = max_resolution
        self.is_moving_average=is_moving_average
        self.netD = Discriminator(
            negative_slope=negative_slope,
            is_spectral_norm=is_spectral_norm
            ).to(device)
        self.netG = Generator(
            n_dims=n_dims,
            max_resolution=max_resolution,
            negative_slope=negative_slope,
            is_spectral_norm=is_spectral_norm
            ).to(device)
        self.optimizer_d = torch.optim.Adam(self.netD.parameters(),lr=d_lr,betas=d_betas)
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(),lr=g_lr,betas=g_betas)
        self.loss = WassersteinGP(self.netD) if loss =="wasserstein" else Hinge()
        if(is_moving_average):
            self.mvag_netG = self.netG(
                n_dims=n_dims,
                max_resolution=max_resolution,
                negative_slope=negative_slope,
                is_spectral_norm=is_spectral_norm
                ).to(device)
            self.ema = EMA()
            self.ema.setup(self.mvag_netG)
             
    def train_d(self,real_img,fake_img):
        self.optimizer_d.zero_grad()
        real = self.netD(real_img)
        fake = self.netD(fake_img)
        loss = self.loss(real,fake,real_img,fake_img)
        loss.backward()
        self.optimizer_d.step()
        return loss 
    
    def train_g(self,fake_img):
        self.optimizer_g.zero_grad()
        fake = self.netD(fake_img)
        loss = -torch.mean(fake)
        loss.backward()
        self.optimizer_g.step()
        if(self.is_moving_average):
            self.ema.apply(self.netG,self.mvag_netG)
        return loss
    
    def train(self,loader,epochs,img_size,save_num,is_tensorboard): 
        for epoch in tqdm(range(epochs),desc="Epochs",total=epochs+self.total_epochs,initial=self.total_epochs,leave=False):
            for step,data in enumerate(tqdm(loader,desc="Steps",leave=False),start=1):
                real_imgs,_ = data
                real_imgs = TF.resize(real_imgs,size=img_size).to(device)
                noise = torch.randn(size=(real_imgs.shape[0],self.n_dims,1,1),device=device)
                fake_imgs = self.netG(noise,True)
                loss_d = self.train_d(real_imgs,fake_imgs).to("cpu")
                self.total_steps += 1
                if(self.total_steps % self.n_dis ==0):
                    loss_g = self.train_g(fake_imgs).to("cpu")
                else:
                    with torch.no_grad():
                        output = self.netD(fake_imgs)
                    loss_g = -torch.mean(output).to("cpu")
                    del output
                if(is_tensorboard):
                    self.log_writer.add_scalars(
                        "Loss",
                        {"Generator":loss_g.item(),"Discriminator":loss_d.item()},
                        global_step=self.total_steps
                        )                    
                if(step % save_num == 0):
                    if(self.is_moving_average):
                        netG = self.mvag_netG
                    else:
                        netG = self.netG
                    with torch.no_grad():
                        sample_img = (netG(self.fixed_noise).detach() + 1) / 2
                        sample_img = sample_img.to("cpu")
                    save_img(sample_img, file_name=f"size_{img_size}epoch{self.total_epochs}_step{step}")
                    for i in range(5):
                        check_vec = torch.randn(size=(1,self.n_dims,1,1),device=device)
                        with torch.no_grad():
                            check_img = (torch.squeeze(netG(check_vec)).detach() + 1) / 2
                            check_img = check_img.to("cpu")
                        save_img(check_img, file_name=f"size_{img_size}epoch{self.total_epochs}_step{step}_check{i}",is_grid=False)
                    else:
                        del check_vec 
            self.total_epochs += 1
            if((epoch+1) % 2 == 0):
                self.save_model(f"params\size{img_size}_epoch_{self.total_epochs-1}")
            self.netD.update()
            self.netG.update()
            if(self.is_moving_average):
                self.mvag_netG.update()

    def fit(self,dataset,epochs,batch_size=10,shuffle=True,num_workers=0,is_tensorboard=True,image_num=10):
        if(is_tensorboard):
            self.log_writer = tensorboard.SummaryWriter(log_dir="./logs")
        self.fixed_noise = torch.randn(size=(16,self.n_dims,1,1),device=device)
        current_size = self.netG.img_size.item()
        begin = int(log2(current_size))
        end = int(log2(self.max_resolution))
        if(not isinstance(epochs,list,tuple)):
            epochs = [epochs for _ in range(begin,end)]
        elif(len(epochs) != (end - begin + 1)):
            epochs += [epochs[-1] for _ in range(end-len(epochs)+2)] 
        for index,img_size in tqdm(enumerate(range(current_size,self.max_resolution,2),initial=current_size)):
            loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
            save_num = len(loader) // image_num
            self.netG.sample_size = len(loader)
            self.netD.sample_size = len(loader)
            self.train(loader,epochs[index],img_size,save_num,is_tensorboard) 

        if(is_tensorboard):
            self.log_writer.close()
    
    def save_model(self, save_path):
        torch.save({"current_size": self.netG.img_size.item()},save_path+"_size.pt")
        if(self.is_moving_average):
            torch.save({"model_state_mvagg":self.mvag_netG.state_dict()},save_path+"_mvagg.pt")
        super().save_model(save_path)

    def load_model(self, load_path):
        param = torch.load(load_path+"_size.pt")
        current_size = param["current_size"]
        begin,end = int(log2(self.netG.img_size.item())),int(log2(current_size))
        for _ in range(begin,end):
            self.netG.update()
            self.netD.update()
            if(self.is_moving_average):
                self.mvag_netG.update() 
        param = torch.load(load_path+"_mvagg.pt")
        if(self.is_moving_average):
            self.mvag_netG.load_state_dict(param["model_state_mvagg"])
        super().load_model(load_path)
        
