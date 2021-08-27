#%%
from gan_utils import DataLoader

# %%
dataset = DataLoader("dq_image",resolutions=128)
# %%
import torch
loader = torch.utils.data.DataLoader(dataset,batch_size=10,shuffle=True)

# %%
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm
for imgs,_ in tqdm(loader):
    imgs = (imgs + 1.0)/2.0
    for img in imgs:
        img = TF.to_pil_image(img)
# %%
