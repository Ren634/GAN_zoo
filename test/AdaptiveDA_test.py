#%%
import numpy
from numpy.lib.function_base import append 
import torch
from gan_modules import AdaptiveDA
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

# %%
ADA = AdaptiveDA()
img1 = torchvision.io.read_image("chrome_512x512.png").unsqueeze(0)
img2 = torchvision.io.read_image("firefox_512x512.png").unsqueeze(0)
imgs = torch.cat([img1,img2])
imgs = imgs /255
# %%
def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x
#%%
ADA._apply_p = 0.5
applyied = ADA.apply(imgs)
applyied = torch.clamp(applyied,min=0,max=1)
for img in applyied:
    img = TF.to_pil_image(img)
    plt.imshow(img)
    plt.show()


# %%
