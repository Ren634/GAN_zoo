#%%
import torch
from LSGAN import Generator

netG = Generator(512,128,0.01,(0,0.999)).to("cuda")
print(netG)
# %%
