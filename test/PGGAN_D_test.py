#%%
import numpy
import torch
from PGGAN import Discriminator
# %%
x = torch.randn(size=(1,3,4,4))
netD = Discriminator(is_spectral_norm=False)

# %%
y = netD(x)
# %%
netD.update()
x = torch.randn(size=(1,3,8,8))
y = netD(x)
#%%
