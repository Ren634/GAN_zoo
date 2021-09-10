import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def hinge(real,fake):
    output = F.relu(1-real).mean() + F.relu(1+fake).mean()
    return output 
    
class Hinge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,real,fake,*imgs):
        return hinge(real,fake)

class WassersteinGP(nn.Module):
    def __init__(self,net,penalty_coeff=10):
        super().__init__()
        self.penalty_coeff = penalty_coeff
        self.net = net

    def forward(self,real,fake,*imgs):
        real_img,fake_img = [img.clone() for img in imgs]
        coeff = torch.tensor(
            np.random.uniform(size=(real.shape[0],1,1,1)),
            requires_grad=True,
            device=real.device,
            dtype=real.dtype
            )
        penalty_input = fake_img * coeff + (1 - coeff)*real_img
        penalty_output = self.net(penalty_input)
        gradient = torch.autograd.grad(penalty_output,penalty_input,create_graph=True)[0]
        penalty = torch.square(torch.norm(gradient,dim=1)-1)
        output = fake.mean() - real.mean() + self.penalty_coeff * penalty.mean()
        return output

        


        
