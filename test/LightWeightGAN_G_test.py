#%%
import numpy as np
import torch
from LightWeightGAN import Generator
#%%
device = "cuda" if torch.cuda.is_available else "cpu"
x = torch.randn((1,256,1,1),device=device)
netG = Generator(256,128).to(device)
y = netG(x)

# %%

# %%
