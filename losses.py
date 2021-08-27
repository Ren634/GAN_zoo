import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def hinge(dx,dgx):
    output = F.relu(1-dx).mean() + F.relu(1+dgx).mean()
    return output


class Wgangp(nn.Module):
    def __init__(self,netD,device,gradient_weights=10,gamma=1.0):
        super().__init__()
        self.netD = netD
        self.device = device
        self.gradient_weights = gradient_weights
        self.gamma = gamma
        self.epsilon_diff = 0.001

    def forward(self,real_img,fake_img,dx,dgx):
        b,_,_,_ = dx.shape
        epsilon = torch.rand(size=(b,1,1,1),device=self.device)
        alpha = torch.autograd.Variable(epsilon * real_img +(1-epsilon)*fake_img,requires_grad=True)
        output = self.netD(alpha)
        gradients = torch.autograd.grad(outputs=output,inputs=alpha,grad_outputs=torch.ones_like(output),retain_graph=True,create_graph=True,only_inputs=True)[0]
        penalty = torch.mean(torch.square(torch.linalg.matrix_norm(gradients,dim=(2,1))-self.gamma)/self.gamma)
        return dx.mean()-dgx.mean()+ penalty * self.gradient_weights + self.epsilon_diff * torch.mean(dx)**2
