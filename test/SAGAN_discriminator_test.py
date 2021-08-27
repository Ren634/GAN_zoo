#%%
import numpy as np
from SAGAN import Discriminator
from torchsummaryX import summary
import torch
# %%
netD = Discriminator().to("cuda")
# %%
summary(netD, torch.zeros(10,3,256,256).to("cuda"))
#%%
