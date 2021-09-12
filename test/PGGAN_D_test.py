#%%
import torch
from PGGAN import Discriminator
# %%
x = torch.randn(size=(1,3,4,4)).to("cuda")
netD = Discriminator(is_spectral_norm=False).to("cuda")

# %%
y = netD(x)
# %%
netD.update()
x = torch.randn(size=(1,3,8,8)).to("cuda")
y = netD(x)
#%%
