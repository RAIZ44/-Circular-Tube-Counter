from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PipeCenterNet(nn.Module):
    """Small U-Net-like CNN producing a dense one-channel center heatmap."""

    output_stride = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBlock(3, 32, stride=2)  # H/2
        self.encoder2 = ConvBlock(32, 64, stride=2)  # H/4
        self.encoder3 = ConvBlock(64, 128, stride=2)  # H/8
        self.bottleneck = ConvBlock(128, 256, stride=2)  # H/16
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        level1 = self.stem(image)
        level2 = self.encoder2(level1)
        level3 = self.encoder3(level2)
        features = self.bottleneck(level3)

        features = F.interpolate(
            features, size=level3.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder3(torch.cat([features, level3], dim=1))
        features = F.interpolate(
            features, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder2(torch.cat([features, level2], dim=1))
        features = F.interpolate(
            features, size=level1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder1(torch.cat([features, level1], dim=1))
        return self.head(features)


def load_checkpoint_model(
    checkpoint_path: str, device: torch.device
) -> tuple[PipeCenterNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PipeCenterNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
