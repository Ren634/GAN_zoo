#%%
import numpy 
import torch
from gan_modules import AdaptiveDA
import torchvision
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from torch.nn import functional as F
#%%
img1 = torchvision.io.read_image("1.jpg").unsqueeze(0)
img2 = torchvision.io.read_image("2.jpg").unsqueeze(0)
imgs = torch.cat([img1,img2])
ADA = AdaptiveDA(50)

#%%
img = ADA.translation(imgs)
plt.imshow(TF.to_pil_image(img[0]))
plt.show()
plt.imshow(TF.to_pil_image(img[1]))
plt.show()
#%%
