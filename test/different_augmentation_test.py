#%%
import numpy as np
import torch
from self_attention import SelfAttention
import torchvision
from different_augmentation import DifferentAugmentation
import matplotlib.pyplot as plt
# %%
x = torchvision.io.read_image("firefox_512x512.png").to("cuda")
x = x.unsqueeze(0)/255.
da = DifferentAugmentation(x.shape[2:],30)
# %%
img = da.apply(x)
img = img.permute(0,2,3,1)
print(img.shape)
# %%
plt.imshow(img[0].cpu())

# %%
