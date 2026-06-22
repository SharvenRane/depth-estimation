"""A small encoder decoder for monocular depth estimation.

The network takes a 3 channel image and predicts a single channel depth map at
the same spatial resolution as the input. It is deliberately compact so it runs
quickly on CPU and trains in a handful of steps on synthetic data.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthNet(nn.Module):
    """U-Net style encoder decoder with skip connections.

    The output is a single channel depth map with the same height and width as
    the input image. The model predicts depth in log space and the head applies
    no activation so values can be any real number, which pairs naturally with
    the scale invariant loss that operates in log space.
    """

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        c = base_channels

        self.enc1 = _conv_block(in_channels, c)
        self.enc2 = _conv_block(c, c * 2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(c * 2, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c * 2, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
