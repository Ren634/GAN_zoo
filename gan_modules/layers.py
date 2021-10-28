import torch
from torch.nn import functional as F
from torch import nn


def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, **kwargs):
    layer = nn.utils.spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            **kwargs)
    )
    return layer


def sn_linear(in_features, out_features, bias=True, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias, **kwargs))


def sn_tconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, **kwargs):
    layer = nn.utils.spectral_norm(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs)
    )
    return layer


class EqualizedLRTConv2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs
        )
        nn.init.normal_(self.weight, mean=0, std=1)
        nn.init.constant_(self.bias, val=0.0)
        if(not isinstance(kernel_size, tuple)):
            kernel_size = (kernel_size, kernel_size)
        f_in = torch.tensor([in_channels, *kernel_size])
        self.scale_factor = torch.sqrt(2 / torch.prod(f_in, dtype=self.weight.dtype))

    def forward(self, inputs):
        inputs *= self.scale_factor
        output = F.conv_transpose2d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation
        )
        return output


class EqualizedLRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs
        )
        nn.init.normal_(self.weight, mean=0, std=1)
        nn.init.constant_(self.bias, val=0.0)
        if(not isinstance(kernel_size, tuple)):
            kernel_size = (kernel_size, kernel_size)
        f_in = torch.prod(torch.tensor([in_channels, *kernel_size], dtype=self.weight.dtype, device=self.weight.device))
        self.scale_factor = torch.sqrt(2 / f_in)

    def forward(self, inputs):
        inputs = self.scale_factor * inputs
        output = F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class EqualizedLRLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs
        )
        nn.init.normal_(self.weight, mean=0, std=1)
        nn.init.constant_(self.bias, val=0.0)
        f_in = torch.tensor(in_features, dtype=self.weight.dtype)
        self.scale_factor = torch.sqrt(2 / f_in)

    def forward(self, inputs):
        inputs = self.scale_factor * inputs
        output = F.linear(inputs, self.weight, self.bias)
        return output


class PixelNorm2d(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        denominator = torch.rsqrt(torch.mean(inputs**2, dim=1, dtype=inputs.dtype, keepdim=True) + self.epsilon)
        output = inputs * denominator
        return output


class MiniBatchStddev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        b, _, h, w = inputs.shape
        std = torch.std(inputs, unbiased=False, dim=0)
        v = torch.mean(std)
        output = torch.cat((inputs, torch.full(size=(b, 1, h, w), fill_value=v.item(), dtype=inputs.dtype, device=inputs.device)), dim=1)
        return output


class GlobalSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.sum(inputs, dim=(-1, -2))


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.f = sn_conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=(1, 1))
        self.g = sn_conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=(1, 1))
        self.h = sn_conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=(1, 1))
        self.v = sn_conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        fx = self.f(x)
        fx = fx.view(b, c // 8, w * h).permute([0, 2, 1])
        gx = self.g(x)
        gx = torch.max_pool2d(gx, kernel_size=(2, 2))
        gx = gx.view(b, c // 8, w * h // 4)
        attention_map = torch.bmm(fx, gx)
        attention_map = F.softmax(attention_map, dim=-1)
        hx = self.h(x)
        hx = torch.max_pool2d(hx, kernel_size=(2, 2))
        hx = hx.view(b, c // 2, w * h // 4).permute([0, 2, 1])
        merged_map = torch.bmm(attention_map, hx)
        merged_map = merged_map.permute([0, 2, 1]).view(b, c // 2, h, w)
        attention_map_v = self.v(merged_map)
        return x + attention_map_v * self.gamma


class ImgToVec(nn.Module):
    def __init__(self, patch_size: int):
        self.axis_patch_size = patch_size**0.5
        assert patch_size / self.axis_patch_size != patch_size, "square root of patch_size must be integer"
        super().__init__()

    def forward(self, inputs):
        _, _, h, w = inputs.shape
        h_patch_size, w_patch_size = int(h / self.axis_patch_size), int(w / self.axis_patch_size)
        patches = [
            inputs[:, :, h_stride:h_patch_size + h_stride, w_stride:w_patch_size + w_stride]
            for h_stride in range(0, h, h_patch_size)
            for w_stride in range(0, w, w_patch_size)
        ]
        flatten = torch.cat([torch.flatten(patch, -3, -1).unsqueeze(dim=1) for patch in patches], dim=1)
        return flatten


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, out_features, heads=8, dims_per_head=64, dropout=0.):
        super().__init__()
        self.root_d = dims_per_head**0.5
        self.heads = heads
        self.dims_per_heads = dims_per_head
        self.to_q = nn.Linear(in_features=in_features, out_features=heads * dims_per_head, bias=False)
        self.to_k = nn.Linear(in_features=in_features, out_features=heads * dims_per_head, bias=False)
        self.to_v = nn.Linear(in_features=in_features, out_features=heads * dims_per_head, bias=False)
        self.main = nn.Sequential(
            nn.Linear(in_features=dims_per_head, out_features=out_features),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        b = inputs.shape[0]
        q = self.to_q(inputs).contiguous().view(b, self.heads, -1, self.dims_per_heads)
        k = self.to_k(inputs).contiguous().view(b, self.heads, -1, self.dims_per_heads)
        v = self.to_v(inputs).contiguous().view(b, self.heads, -1, self.dims_per_heads)
        attention_map = torch.softmax(torch.bmm(q, k.permute(0, 1, 3, 2)), dim=-1) / self.root_d
        attention_map = torch.bmm(attention_map, v)
        output = self.main(attention_map)
        return output
