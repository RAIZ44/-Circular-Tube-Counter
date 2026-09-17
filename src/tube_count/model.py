from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tube_count.targets import center_focal_loss, radius_loss


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TubeNet(nn.Module):
    """Small CenterNet-style detector: center heatmap + normalized radius head.

    Input is RGB in [0, 1], shape (N, 3, S, S). Heads are at resolution S/stride.
    """

    def __init__(self, base_channels: int = 32, stride: int = 4) -> None:
        super().__init__()
        if stride != 4:
            raise ValueError("This backbone is wired for stride=4")
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 4
        self.stem = nn.Sequential(ConvBNAct(3, c1, stride=2), ConvBNAct(c1, c1))
        self.down2 = nn.Sequential(ConvBNAct(c1, c2, stride=2), ConvBNAct(c2, c2))
        self.down3 = nn.Sequential(ConvBNAct(c2, c3, stride=2), ConvBNAct(c3, c3))
        self.up = nn.Sequential(ConvBNAct(c3, c4), ConvBNAct(c4, c4))
        self.skip = ConvBNAct(c2, c4)
        self.heatmap = nn.Sequential(
            nn.Conv2d(c4, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, 1, kernel_size=1),
        )
        self.radius = nn.Sequential(
            nn.Conv2d(c4, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, 1, kernel_size=1),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.constant_(self.heatmap[-1].bias, -2.0)
        nn.init.constant_(self.radius[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        up = F.interpolate(s3, size=s2.shape[-2:], mode="nearest")
        feat = self.up(up) + self.skip(s2)
        heatmap = torch.sigmoid(self.heatmap(feat))
        radius = F.softplus(self.radius(feat))
        return heatmap, radius


def detection_loss(
    pred_hm: torch.Tensor,
    pred_rad: torch.Tensor,
    gt_hm: torch.Tensor,
    gt_rad: torch.Tensor,
    rad_mask: torch.Tensor,
    radius_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    hm = center_focal_loss(pred_hm, gt_hm)
    rad = radius_loss(pred_rad, gt_rad, rad_mask)
    total = hm + radius_weight * rad
    parts = {"loss": float(total.detach()), "hm_loss": float(hm.detach()), "rad_loss": float(rad.detach())}
    return total, parts
