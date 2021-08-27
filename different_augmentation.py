import numpy as np
import torch
import random 
from torch import nn
import torchvision.transforms.functional as TF

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
        


        



