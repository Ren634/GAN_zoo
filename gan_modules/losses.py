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
    def __init__(self,net,penalty_coef=10):
        super().__init__()
        self.penalty_coef = penalty_coef
        self.net = net

    def forward(self,real,fake,*imgs):
        real_img,fake_img = imgs
        batch_size = real.shape[0]
        coef = torch.tensor(
            np.random.uniform(size=(batch_size,1,1,1)),
            requires_grad=True,
            device=real.device,
            dtype=real.dtype
            )
        penalty_input = real_img * coef + (1 - coef)*fake_img
        penalty_output = self.net(penalty_input)
        gradient = torch.autograd.grad(
            penalty_output,
            penalty_input,
            grad_outputs=torch.ones_like(penalty_output),
            create_graph=True,
            retain_graph=True
            )[0]
        penalty = gradient.view(batch_size,-1)
        penalty = torch.square(torch.norm(gradient,dim=1)-1)
        output = fake.mean() - real.mean() + self.penalty_coef * penalty.mean()
        return output

        


        
