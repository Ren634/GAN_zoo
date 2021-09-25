#%%
from PGGAN import Generator
import torch
# %%
netG = Generator(n_dims=512,max_resolution=128,is_spectral_norm=False).to("cuda")
x = torch.randn((1,512,1,1)).to("cuda")
# %%
netG.update()
# %%
x = torch.randn((1,512,1,1)).to("cuda")
x = netG(x)
# %%
