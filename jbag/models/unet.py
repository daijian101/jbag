import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(num_features=mid_channels)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(num_features=out_channels)
        relu2 = nn.ReLU(inplace=True)
        super().__init__(conv1, bn1, relu1, conv2, bn2, relu2)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, width_factor=64, blocks=5):
        super().__init__()
        channels = [width_factor << i for i in range(blocks)]

        block_0 = DoubleConv(in_channels=in_channels, out_channels=channels[0])
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels=channels[i - 1], out_channels=channels[i]))
            for i in range(1, blocks)])

        self.blocks.insert(0, block_0)
        self.out_channels = channels

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()

        in_channels = encoder_channels[::-1]

        self.blocks = nn.ModuleList([
            Up(in_channels=in_channels[i], out_channels=in_channels[i + 1],
               skip_conn_channels=in_channels[i + 1])
            for i in range(0, len(in_channels) - 1)])

        self.out_channels = in_channels[-1]

    def forward(self, x):
        skip_connections = x[-2::-1]
        x = x[-1]
        for i, skip_connection in enumerate(skip_connections):
            x = self.blocks[i](x, skip_connection)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conn_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_conn_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.up(x)
        diff_y = skip_features.size()[2] - x.size()[2]
        diff_x = skip_features.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip_features, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width_factor=64, blocks=5, normalize=True):
        """
        UNet.

        Args:
            in_channels (int, optional, default=1):
            out_channels (int, optional, default=1):
            width_factor (int, optional, default=64):
            blocks (int, optional, default=5):
            normalize (bool, optional, default=True): If `True`, normalize the output using `torch.sigmoid` for one
                dimension output, or `torch.softmax` for multiple classes output.
        """

        super().__init__()
        self.out_channels = out_channels
        encoder = UNetEncoder(in_channels=in_channels, width_factor=width_factor, blocks=blocks)
        decoder = UNetDecoder(encoder_channels=encoder.out_channels)
        decoder_head = nn.Conv2d(in_channels=decoder.out_channels, out_channels=out_channels, kernel_size=1)
        self.unet = nn.Sequential(encoder,
                                  decoder,
                                  decoder_head)

    def forward(self, x):
        x = self.unet(x)
        return x
