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
        imgs = torchvision.utils.make_grid(imgs,nrow=(len(imgs)//4),padding=10)
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
        torch.save(
            {
                "model_state":self.netD.state_dict(),
                "optim_state":self.optimizer_d.state_dict()
            },
            save_path+"_d.pt"
            )
        torch.save(
            {
                "model_state":self.netG.state_dict(),
                "optim_state":self.optimizer_g.state_dict()
            },
            save_path+"_g.pt")
        torch.save(
            {
                "total_epochs":self.total_epochs,
                "total_steps":self.total_steps,
                "fixed_noise":self.fixed_noise
            },
            save_path+"_trainer.pt"
        )

    def load_model(self,load_path):
        load_path_d = load_path+"_d.pt"
        load_path_g = load_path+"_g.pt"
        load_path_trainer = load_path+"_trainer.pt"
        param_d = torch.load(load_path_d)
        param_g = torch.load(load_path_g)
        trainer_state = torch.load(load_path_trainer)
        self.netD.load_state_dict(param_d["model_state"])
        self.netG.load_state_dict(param_g["model_state"])
        self.optimizer_d.load_state_dict(param_d["optim_state"])
        self.optimizer_g.load_state_dict(param_g["optim_state"])
        self.total_steps = trainer_state["total_steps"]
        self.total_epochs = trainer_state["total_epochs"]
        self.fixed_noise = trainer_state["fixed_noise"]
        
def translation(self,inputs):
    b,_,h,w = inputs.shape
    y,x = torch.randint(-h//5,h//5,size=(b,1,1)),torch.randint(-w//5,w//5,size=(b,1,1))
    index_b,index_y,index_x = torch.meshgrid(
        torch.arange(b,dtype=torch.long),
        torch.arange(h,dtype=torch.long),
        torch.arange(w,dtype=torch.long)
    )
    index_y = torch.clamp(index_y + y , min=0, max=h+1)
    index_x = torch.clamp(index_x + x , min=0, max=w+1)
    inputs = F.pad(inputs,[1,1,1,1,0,0,0,0]) # dim -1(left,right) dim -2(left, right) # dim -3(left, right) -4(left, right)
    return inputs.permute(0,2,3,1).contiguous()[index_b,index_y,index_x].permute(0,3,1,2).contiguous()
            
def cutout(self,inputs):
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

class AdaptiveDA: 
    def __init__(self,limit,frequency=4,threshold=0.6,const=0.05):
        self.apply_p = 0
        self.threshold = threshold
        self.const = 0.05
        self.limit = limit
        self.frequency = frequency
        self.n = 0
        self.total_n = 0
        
    def adjust_p(self,inputs):
        rt = torch.sign(inputs).mean()
        if(self.apply_p < 1 and rt>self.threshold):
            self.apply_p += self.const 
        if(self.apply_p > 0 and rt<self.sh):
            self.apply_p -= self.const

    def apply(self,inputs,target=True):
        if(self.n == self.frequency and self.total_n < self.limit):
            self.adjust_p(inputs)
            
        

        

        
        

        


        