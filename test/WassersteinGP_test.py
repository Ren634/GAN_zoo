#%%
from PGGAN import Discriminator 
import torch
from gan_modules import WassersteinGP
device = "cuda" if torch.cuda.is_available else "cpu"
netD = Discriminator(is_spectral_norm=False).to(device)
real_img = torch.randn((1,3,4,4),device=device)
x = netD(real_img)
for _ in range(2,7):
    netD.update()
real_img = torch.randn((1,3,128,128),device=device)
fake_img = torch.randn((1,3,128,128),device=device)
#%%
loss_fn = WassersteinGP(netD)
#%%
real = netD(real_img)
fake = netD(fake_img)
loss = loss_fn(real,fake,real_img,fake_img)
#%%
