#%%
import numpy as np
import torch
from LightWeightGAN import Generator
#%%
resolution = 128
device = "cuda" if torch.cuda.is_available else "cpu"
netG = Generator(100,resolution).to(device)
#%%
x = torch.randn((1,100,1,1),device=device)
for _ in range(5):
    y = netG(x)

# %%

# %%
