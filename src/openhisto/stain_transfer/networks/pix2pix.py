#!/usr/bin/env python3
import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        x = torch.cat((x, skip), dim=1)
        return x

class UNetGenerator(nn.Module):
    """U-Net generator from pix2pix paper: 8 down/ups with skip connections."""
    def __init__(self, in_ch=3, out_ch=3, ngf=64, use_dropout=True):
        super().__init__()
        # Downsampling
        self.d1 = UNetDown(in_ch, ngf, normalize=False)   # 256->128
        self.d2 = UNetDown(ngf, ngf * 2)                      # 128->64
        self.d3 = UNetDown(ngf * 2, ngf * 4)                     # 64->32
        self.d4 = UNetDown(ngf * 4, ngf * 8)                     # 32->16
        self.d5 = UNetDown(ngf * 8, ngf * 8)                     # 16->8
        self.d6 = UNetDown(ngf * 8, ngf * 8)                     # 8->4
        self.d7 = UNetDown(ngf * 8, ngf * 8)                     # 4->2
        self.d8 = UNetDown(ngf * 8, ngf * 8, normalize=False)    # 2->1

        # Upsampling
        self.u1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5 if use_dropout else 0.0)
        self.u2 = UNetUp(ngf * 16, ngf * 8, dropout=0.5 if use_dropout else 0.0)
        self.u3 = UNetUp(ngf * 16, ngf * 8, dropout=0.5 if use_dropout else 0.0)
        self.u4 = UNetUp(ngf * 16, ngf * 8)
        self.u5 = UNetUp(ngf * 16, ngf * 4)
        self.u6 = UNetUp(ngf * 8, ngf * 2)
        self.u7 = UNetUp(ngf * 4, ngf)
        self.u8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        out = self.u8(u7)
        return out

class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN"""
    def __init__(self, in_ch=3, out_ch=3, ndf=64):
        super().__init__()
        # Input is concatenation of condition (A) and target/generated (B): 3+3 channels
        c_in = in_ch + out_ch
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(c_in, ndf, norm=False),          # 256->128
            *block(ndf, ndf * 2),                   # 128->64
            *block(ndf * 2, ndf * 4),               # 64->32
            nn.Conv2d(ndf * 4, 1, 4, 1, 1)          # 32->31 (Patch map)
        )

    def forward(self, cond, x):
        # cond = A, x = B or G(A)
        return self.model(torch.cat([cond, x], dim=1))