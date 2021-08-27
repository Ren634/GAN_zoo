#%%
import numpy as np
from SAGAN import Generator
from torchsummaryX import summary
import torch
# %%
netG = Generator().to("cuda")
summary(netG, torch.zeros(10,512,1,1).to("cuda"))
# %%
