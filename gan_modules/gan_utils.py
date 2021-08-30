import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF
import datetime
import glob
import os
import random

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
        
class DifferentAugmentation: 
    def __init__(self,img_shape,trans_range=30,apply_p=0.85):
        self.h ,self.w = img_shape
        self.trans_range = trans_range
        self.device = "cuda" if(torch.cuda.is_available()) else "cpu"
        self.apply_p = apply_p

    def apply(self,inputs,target=True):
        if(target):
            self.method_num = np.random.choice(range(9),p=[self.apply_p]+[(1-self.apply_p)/8]*8)

        if(self.method_num==0):
            return inputs

        elif(self.method_num==1):
            if(target):
                self.brightness_factor = random.uniform(1.1, 3)
            applied = TF.adjust_brightness(inputs, brightness_factor=self.brightness_factor)

        elif(self.method_num==2):
            applied = TF.hflip(inputs)

        elif(self.method_num==3):
            applied = TF.vflip(inputs)

        # TranslationX
        elif(self.method_num==4):
            if(target):
                self.translate_x = [random.randint(-self.trans_range,self.trans_range),0]
            applied = TF.affine(inputs, angle=0.0, translate=self.translate_x, scale=1.0, shear=0.0)
            if(self.translate_x[0] < 0):
                translate = -self.translate_x[0]
                applied[:,:,:,self.w - translate:] = inputs[:,:,:,:translate]
            elif(self.translate_x == 0):
                applied = inputs
            else:
                applied[:,:,:,:self.translate_x[0]] = inputs[:,:,:,self.w - self.translate_x[0]:]
        
        # TranslationY
        elif(self.method_num==5):
            if(target):
                self.translate_y = [0,random.randint(-self.trans_range,self.trans_range)]
            applied = TF.affine(inputs, angle=0.0, translate=self.translate_y, scale=1.0, shear=0.0)
            if(self.translate_y[1] < 0):
                translate = -self.translate_y[1]
                applied[:,:,self.h - translate:,:] = inputs[:,:,:translate,:]
            elif(self.translate_y ==0):
                applied = inputs
            else:
                applied[:,:,:self.translate_y[1],:] = inputs[:,:,self.h - self.translate_y[1]:,:]

        # CutOut
        elif(self.method_num == 6):
            if(target):
                self.x = random.randint((self.w-1)//2 - 30,30+(self.w-1)//2)
                self.y = random.randint((self.h-1)//2 - 30,30+(self.h-1)//2)
                self.boxsize = random.randint(10,30)
            applied = TF.erase(inputs, self.x, self.y, self.boxsize, self.boxsize, 0)
        
        # change brightness along channels
        elif(self.method_num==7):
            if(target):
                c = random.uniform(0,0.25)
                self.channel_brightness_factor = torch.zeros(inputs.shape[1:],dtype=inputs.dtype).to(self.device)
                self.channels = random.randint(0,2)
                self.channel_brightness_factor[self.channels,:,:] += c
            applied = inputs + self.channel_brightness_factor
            if(not target):
                del self.channel_brightness_factor
                torch.cuda.empty_cache()
        
        # add noise
        elif(self.method_num ==8):
            if(target):
                self.noise = torch.randn(size=inputs.shape[1:]).to(self.device) / 10
            applied = inputs + self.noise
            if(not target):
                del self.noise
                torch.cuda.empty_cache()
        return applied
