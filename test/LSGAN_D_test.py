#%%
import torch
from LSGAN import Discriminator

netD = Discriminator(128,0.01,(0,0.999)).to("cuda")
print(netD)
# %%
x = torch.randn(1,3,128,128).to("cuda")
y = netD(x)

# %%
