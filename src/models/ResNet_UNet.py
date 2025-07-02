import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    A basic residual block for ResNet with two 3×3 conv layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolution.
        downsample (nn.Module or None): Optional downsampling layer for the skip connection.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNetEncoder(nn.Module):
    """
    Custom ResNet34-like encoder built using BasicBlock, without external dependencies.

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # Initial conv layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64,  64,  3)
        self.layer2 = self._make_layer(64,  128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class ResNetUNet(nn.Module):
    """
    U-Net using a custom ResNetEncoder implemented with only torch and torch.nn.

    Args:
        in_channels (int): Number of input image channels.
        num_classes (int): Number of output segmentation classes.
        base_filters (int): Number of filters in the bottleneck layer.
    """
    def __init__(self, in_channels=1, num_classes=3, base_filters=256):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Decoder upsample + DoubleConv
        self.up3 = nn.ConvTranspose2d(base_filters, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128+128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        b = self.bottleneck(x4)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        return self.head(d0)

# Example usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNetUNet(in_channels=1, num_classes=3).to(device)
