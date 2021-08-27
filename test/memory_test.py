#%%
import numpy as np
import torch
from SAGAN import Generator,Discriminator
# %%
if (torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"
netG = Generator().to(device)
netD = Discriminator().to(device)

# %%
x = torch.zeros(10,512,1,1).to(device)
img = netG(x)
loss = netD(img)


# %%
print(img.shape,loss.shape)

# %%
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a 
print("total_memory: ",t)
print("reserved_memory: ",r)
print("allocated_memory: ",a)
print("free inside reserved: ",f)

#%%
