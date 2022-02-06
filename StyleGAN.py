import torch
from torch import nn
from gan_modules import *


class MappingNet(nn.Module):
    def __init__(self, in_features=512, out_features=512, negative_slope=0.01, is_inplace=True):
        super().__init__()
        # Linear to Conv2d
        self.main = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=is_inplace),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class AdaIN(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.to_ysi = EqualizedLRConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3))
        self.to_ybi = EqualizedLRConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3))

    def forward(self, x, y):
        ysi, ybi = self.to_ysi(y), self.to_ybi(y)
        mean = torch.mean(x, dim=1)
        std = torch.std(x)
        normalized = (x - mean) / std
        output = ysi * normalized + ybi
        return output


class RGBAdd(nn.Module):
    def __init__(self, sample_size):
        super().__init__()
        self.alpha = torch.tensor(1.)
        self.const = torch.tensor(1 / sample_size)
        self.register_buffer("coef", self.alpha)
        self.register_buffer("additional factor", self.const)

    def forward(self, RGBs):
        if(len(RGBs) == 2):
            RGB, old_RGB = RGBs
        else:
            return RGBs[0]
        output = (1 - self.alpha) * old_RGB + self.alpha * RGB
        return output


class BlockG(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01, upsample="nearest"):
        super().__init__()
        if(upsample != "nearest"):
            upsampling = EqualizedLRTConv2d
            augment = {"in_channels": in_channels, "out_channels": in_channels, "kernel_size": 2, "stride": 2}
        else:
            upsampling = nn.Upsample
            augment = {"scale_factor": 2, "mode": "nearest"}
        self.main = nn.ModuleList([
            nn.Sequential(
                upsampling(*augment),
                EqualizedLRConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)),
                nn.LeakyReLU(negative_slope=negative_slope),
                PixelNorm2d(epsilon=1e-8)
            ),
            nn.Sequential(
                EqualizedLRConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3)),
                nn.LeakyReLU(negative_slope=negative_slope),
                PixelNorm2d(epsilon=1e-8)
            ),
        ])
        self.adains = nn.ModuleList([AdaIN(out_channels=out_channels), AdaIN(out_channels=out_channels)])

    def upsampling_layer(self):
        return self.main[0][0]

    def forward(self, x, latent, noise):
        for layer, adain in enumerate(self.main, self.adains):
            x = layer(x) + noise
            x = adain(x, latent)
        return x


class FirstBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        super().__init__()
        self.main = nn.Sequential(
            EqualizedLRConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.pixel_norm = PixelNorm2d()
        self.adains = nn.ModuleList([AdaIN(out_channels=out_channels), AdaIN(out_channels=out_channels)])

    def forward(self, x, latent, noise):
        x = self.pixel_norm(x) + noise
        x = self.adains[0](x, latent)
        x = self.main(x)
        output = self.adains[1](x, latent)
        return output


class Generator(nn.Module):
    def __init__(self, n_dims, batch_size, negative_slope=0.01, upsample="nearest"):
        super().__init__()
        self.const = torch.randn(size=(batch_size, 512, 4, 4))
        self.latent2style = nn.Linear(in_features=n_dims, out_features=n_dims)
        self.img_size = torch.tensor(4)
        self.negative_slope = negative_slope
        self.upsample = upsample
        self.register_buffer("img_size", self.img_size)
        self.register_buffer("const", self.const)
        self.__sample_size = 1
        self.out_channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }
        self.synthesis_network = nn.ModuleList([
            FirstBlockG(in_channels=512, out_channels=self.out_channels[4], negative_slope=negative_slope)
        ])
        self.to_rgb = nn.ModuleDict({
            "up_to_date": EqualizedLRConv2d(in_channels=self.out_channels[4], out_channels=3, kernel_size=(1, 1))
        })
        self.output_layer = RGBAdd(self.__sample_size)

    def update(self):
        img_size = self.img_size.item()
        self.synthesis_network.append(
            BlockG(
                in_channels=self.out_channels[img_size],
                out_channels=self.out_channels[img_size * 2],
                negative_slope=self.negative_slope,
                upsample=self.upsample).to(device)
        )
        self.to_rgb["old"] = self.to_rgb["up_to_date"]
        self.to_rgb["up_to_date"] = EqualizedLRConv2d(
            in_channels=self.out_channels[img_size * 2],
            out_channels=3,
            kernel_size=(1, 1)
        ).to(device)
        self.img_size *= 2

    def forward(self, x):
        RGBs = []
        for layer in self.synthesis_network:
            x = layer(x)
            if(self.img_size // 2 == x.shape[-1]):
                upsampled_x = self.synthesis_network[-1].upsampling_layer(x)
                RGBs.insert(0, self.to_rgb["old"](upsampled_x))
        RGBs.append(self.to_rgb["up_to_date"](x))
        output = self.output_layer(RGBs)
        return output
