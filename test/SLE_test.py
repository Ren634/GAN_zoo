#%%
import numpy as np
import torch
from LightWeightGAN import SLE
#%%
device = "cuda" if torch.cuda.is_available else "cpu"
sle = SLE(512,64).to(device)
x = torch.randn((1,512,8,8),device=device)
shutcut = torch.randn((1,64,128,128),device=device)
y = sle(x,shutcut)

# %%
