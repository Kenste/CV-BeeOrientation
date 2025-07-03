import torch
import torch.nn as nn
from torchvision import models

from src.models.base import DoubleConv


class ResUNet18(nn.Module):
    """
    U-Net with a ResNet-18 encoder backbone, adapted from
    https://github.com/qubvel/segmentation_models.pytorch

    Architecture:
      - Encoder: ResNet-18 pretrained on ImageNet
      - Decoder: 4 upsampling blocks with skip connections
      - Output: per-pixel class logits

    Args:
        num_classes (int): number of output classes
        dropout (float): dropout probability in decoder

    Input:  (B, 1, H, W) grayscale image
    Output: (B, num_classes, H, W) logits
    """
    def __init__(self, num_classes=3, dropout=0.0):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # --- Input adapter ---
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)  # 1 → 3

        # --- Encoder ---
        self.enc0 = nn.Sequential(  # Stem block: 3 → 64
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool0 = resnet.maxpool

        self.enc1 = resnet.layer1  # 64 → 64
        self.enc2 = resnet.layer2  # 64 → 128
        self.enc3 = resnet.layer3  # 128 → 256
        self.enc4 = resnet.layer4  # 256 → 512

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256, dropout)  # (256+256) → 256

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128, dropout)  # (128+128) → 128

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64, dropout)  # (64+64) → 64

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(96, 32, dropout)  # (32+64) → 32

        # --- Final output ---
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)  # 32 → 3

    def forward(self, x):
        # Encoder
        x = self.input_adapter(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder
        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        # Output
        out = self.final_up(d1)
        return self.head(out)