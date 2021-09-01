#%%
import numpy as np
import torch
from torch._C import device
from LightWeightGAN import Discriminator

# %%
device = "cuda" if torch.cuda.is_available else "cpu"
netD = Discriminator(1024).to(device)
x = torch.randn(1,3,1024,1024,device=device)
y,recon = netD(x)
#%%
recon_16,recon8 = recon

# %%
