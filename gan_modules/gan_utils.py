import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import torchvision
import torchvision.transforms.functional as TF
import datetime
import glob
import os

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
        imgs = imgs.squeeze(imgs)
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
        self.file_paths = path

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        label = 1
        img = torchvision.io.read_image(self.file_names[index])
        img = TF.resize(img, size=self.resolutions)
        img = (img - 127.5)/127.5
        return img,label

class EMA:
    def __init__(self,weight_decay=0.995):
        self.weight_decay= weight_decay
        self.__iter_counter = 0

    def setup(self,mvag_net):
        for params in mvag_net.parameters():
            params.requires_grad = False

    @property
    def iter_counter(self):
        return  self.__iter_counter

    @iter_counter.setter
    def iter_counter(self,values):
        self.__iter_counter = values

    def apply(self,net,mvag_net):
        beta = min(1-(1/(self.__iter_counter+1)),self.weight_decay)
        for params,mvag_params in zip(net.parameters(),mvag_net.parameters()):
            mvag_params.data = mvag_params.data* beta + (1-beta)*params.data
        self.iter_counter += 1

class GAN:

    def __init__(self):
        self.total_epochs = 0
        self.total_steps = 0
        self.params = {}
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
        self.params.update(params)
        if(save_path[-3:] != ".pt"):
            save_path = save_path + ".pt"
        torch.save(params,save_path)

    def load_model(self,load_path):
        if(save_path[-3:] != ".pt"):
            save_path = save_path + ".pt"
        params = torch.load(load_path+".pt")
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

class AdaptiveDA(nn.Module):
    def __init__(self,frequency=4,threshold=0.6):
        self.prob = torch.tensor(0)
        self.threshold = torch.tensor(threshold)
        self.frequency = torch.tensor(frequency)
        self.n = torch.tensor(0)
        self.functions = [
            random_brightness,
            random_contrast,
            random_saturation,
            random_translation,
        ]
        self._epoch = 0
        self._batch_size = 0

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self,value):
        self._batch_size = value

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self,value):
        self._epoch = value

    def adjust_p(self,real_logit):
        rt = torch.sign(real_logit).mean()
        self.prob += torch.sign(rt - self.threshold)*(self.epoch*self.batch_size)/(self.batch_size*1000)
        self.prob = torch.clamp(self.prob,min=0,max=1)

    def forward(self,x,target=False):
        if(target):
            self.n += 1
            if(self.n == self.frequency):
                self.adjust_p(x)
                self.n = 0
        prob = self.prob.item()
        is_applying_list = np.random.choice(2,size=len(self.functions),p=[prob,1-prob])
        for function,is_applying in enumerate(self.functions,is_applying_list):
            if(is_applying):
                x = function(x)
        return x
