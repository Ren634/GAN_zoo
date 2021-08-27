#%%
import numpy as np
import torch 
from gan_utils import MiniBatchStddev
#%%
mbs = MiniBatchStddev()
x = torch.randn((10,128,32,32),device="cuda")
y = mbs(x)

# %%
