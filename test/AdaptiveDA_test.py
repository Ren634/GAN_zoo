#%%
import matplotlib
import numpy 
import torch
from gan_modules import AdaptiveDA
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

# %%
ADA = AdaptiveDA(limit=50)
img1 = torchvision.io.read_image("a00001-0.jpg").unsqueeze(0)
img2 = torchvision.io.read_image("a00002-0.jpg").unsqueeze(0)
imgs = torch.cat([img1,img2])
#%%
# %%
cutout1 = ADA.cutout(imgs)
for img in cutout1:
    img = TF.to_pil_image(img)
    plt.imshow(img)
    plt.show()

# %%
