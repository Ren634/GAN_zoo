import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF
import datetime
import glob
import os

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
        