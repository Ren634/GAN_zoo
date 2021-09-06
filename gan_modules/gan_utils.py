import numpy as np
import torch
from torch._C import dtype
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF
import datetime
import glob
import os
import random

device = "cuda" if torch.cuda.is_available else "cpu"

def save_img(imgs,file_name,img_format="png",is_grid=True):
    today = datetime.date.today().strftime("%Y-%m-%d")
    if(is_grid):
        imgs = torchvision.utils.make_grid(imgs,nrow=(len(imgs)//4))
        file_path = f"./logs/imgs/{today}_{file_name}."+img_format
        os.makedirs("./logs/imgs",exist_ok=True) 
    else:
        file_path = f"./logs/check_imgs/{today}_{file_name}."+img_format
        os.makedirs("./logs/check_imgs",exist_ok=True)
    imgs = TF.to_pil_image(imgs)
    imgs.save(file_path)

class DataLoader(torch.utils.data.Dataset):
    def __init__(self,path,resolutions,data_format="jpg"):
        super().__init__()
        if(path[-1]=='/'):
            file_paths = path +"*."+data_format
        else:
            file_paths = path +"/*."+data_format
        self.resolutions = resolutions     
        self.file_names = glob.glob(file_paths,recursive=True)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        label = 1
        img = torchvision.io.read_image(self.file_names[index])
        img = TF.resize(img, size=self.resolutions)
        img = (img - 127.5)/127.5
        return img,label
      
class GAN:

    def __init__(self):
        self.total_epochs = 0
        self.total_steps = 0
    os.makedirs(f"./params",exist_ok=True)

    def save_model(self,save_path):
        params = {
                "model_state_d":self.netD.state_dict(),
                "optim_state_d":self.optimizer_d.state_dict(),
                "model_state_g":self.netG.state_dict(),
                "optim_state_g":self.optimizer_g.state_dict(),
                "total_epochs":self.total_epochs,
                "total_steps":self.total_steps,
                "fixed_noise":self.fixed_noise, 
        }
        if(save_path[-3:] != ".pt"):
            save_path = save_path + ".pt"
        torch.save(params,save_path)

    def load_model(self,load_path):
        params = torch.load(load_path)
        self.netD.load_state_dict(params["model_state_d"])
        self.netG.load_state_dict(params["model_state_g"])
        self.optimizer_d.load_state_dict(params["optim_state_d"])
        self.optimizer_g.load_state_dict(params["optim_state_g"])
        self.total_steps = params["total_steps"]
        self.total_epochs = params["total_epochs"]
        self.fixed_noise = params["fixed_noise"]
        
def random_translation(inputs):
    b,_,h,w = inputs.shape
    y,x = torch.randint(-h//8,h//8,size=(b,1,1)),torch.randint(-w//8,w//8,size=(b,1,1))
    index_b,index_y,index_x = torch.meshgrid(
        torch.arange(b,dtype=torch.long),
        torch.arange(h,dtype=torch.long),
        torch.arange(w,dtype=torch.long)
    )
    index_y = torch.clamp(index_y + y , min=0, max=h+1)
    index_x = torch.clamp(index_x + x , min=0, max=w+1)
    inputs = F.pad(inputs,[1,1,1,1,0,0,0,0]) # dim -1(left,right) dim -2(left, right) # dim -3(left, right) -4(left, right)
    return inputs.permute(0,2,3,1).contiguous()[index_b,index_y,index_x].permute(0,3,1,2).contiguous()
            
def random_cutout(inputs):
    b,_,h,w = inputs.shape 
    y,x= torch.randint(10,h//3,size=(1,)),torch.randint(10,w//3,size=(1,))
    index_b,index_y,index_x = torch.meshgrid(
        torch.arange(b,dtype=torch.long),
        torch.arange(y.item(),dtype=torch.long),
        torch.arange(x.item(),dtype=torch.long)
    )
    index_y = torch.clamp(index_y + torch.randint(0,h-10,size=(b,1,1)),min=0,max=h-1)
    index_x = torch.clamp(index_x + torch.randint(0,w-10,size=(b,1,1)),min=0,max=w-1)
    mask = torch.ones((b,h,w),device=inputs.device,dtype=inputs.dtype)
    mask[index_b,index_y,index_x] *= 0
    return inputs * mask.unsqueeze(1)

def random_saturation(inputs):
    mean = torch.mean(inputs,dim=1,keepdim=True,dtype=inputs.dtype)
    factor = torch.rand(size=(inputs.shape[0],1,1,1),dtype=inputs.dtype,device=inputs.device) * 2
    return (inputs - mean) * factor + mean

def random_brightness(inputs):
    factor = torch.rand(size=(inputs.shape[0],1,1,1),dtype=inputs.dtype,device=inputs.device) / 4
    return inputs + factor

def random_contrast(inputs):
    b,c,h,w = inputs.shape
    mean = torch.mean(inputs,dim=(1,2,3),keepdim=True)
    factor = torch.rand(size=(b,1,1,1),dtype=inputs.dtype,device=inputs.device) + 0.5
    return (inputs - mean) * factor + mean

class AdaptiveDA: 
    def __init__(self,net,frequency=4,threshold=0.6,const=0.01):
        self._apply_p = 0
        self.threshold = threshold
        self.const = const
        self.frequency = frequency
        self.n = 0
        self.functions = [
            random_brightness,
            random_contrast,
            random_saturation,
            random_translation,
        ]
        self.netD = net
        self.rt = 0
        
    def adjust_p(self,x):
        with torch.no_grad():
            x = self.netD(x)
        self.rt = torch.sign(x).mean().item()
        if(self._apply_p < 1 and self.rt>self.threshold):
            self._apply_p = min(self._apply_p + self.const,0.5)
        if(self._apply_p > 0 and self.rt<self.threshold):
            self._apply_p = max(self._apply_p - self.const,0)

    def apply(self,x,target=False):
        if(target):
            self.n += 1
            if(self.n == self.frequency):
                self.adjust_p(x)
                self.n = 0
        if(np.random.choice([True,False],p=[self._apply_p,1-self._apply_p])):
            for function in self.functions:
                    x = function(x)
        return x
            
        
            
            
        

        

        
        

        


        