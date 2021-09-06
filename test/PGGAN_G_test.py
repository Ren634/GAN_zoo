#%%
from PGGAN import Generator
import numpy
import torch
# %%
netG = Generator(n_dims=512,max_resolution=128,is_spectral_norm=False)
x = torch.randn((1,512,1,1))
# %%
x = netG(x)
# %%
netG.update()

# %%
x = torch.randn((1,512,1,1))
x = netG(x)
# %%
