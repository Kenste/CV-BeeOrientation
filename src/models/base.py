import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv → ReLU) ×2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3(nn.Module):
    """
    3-level U-Net model for bee head/tail segmentation, based on the architecture from Bozek et al.'s "Markerless
    tracking of an entire honey bee colony" (https://www.nature.com/articles/s41467-021-21769-1).

    Architecture:
      - Encoder: 3 blocks with 32 → 64 → 128 filters
      - Bottleneck: 256 filters
      - Decoder: 3 blocks with 128 → 64 → 32 filters
      - Output head: 3-class segmentation (background, head, tail)

    Input:
      1×160×160 grayscale image

    Output:
      3×160×160 per-pixel logits
    """

    def __init__(self, in_channels=1, num_classes=3, base_filters=32):
        super().__init__()
        # --- Encoder ---
        self.enc1 = DoubleConv(in_channels, base_filters)  # 1 → 32
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_filters, base_filters * 2)  # 32 → 64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)  # 64 → 128
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck (middle) ---
        self.bottleneck = DoubleConv(base_filters * 4, base_filters * 8)  # 128 → 256

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)  # (128+128) → 128

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)  # (64+64) → 64

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)  # (32+32) → 32

        # --- Final 1×1 conv to classes ---
        self.head = nn.Conv2d(base_filters, num_classes, kernel_size=1)  # 32 → 3

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Output
        return self.head(d1)
