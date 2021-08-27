#%%
import numpy as np
import torch
from gan_utils import save_img

# %%
x = torch.randn(14,3,128,128)

# %%
save_img(x,"hogehoge","png")
#%%
