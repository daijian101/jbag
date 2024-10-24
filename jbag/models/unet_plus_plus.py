from typing import Union, Type

import torch
import torch.nn as nn

from jbag.config import Config
from jbag.models.network_weight_initialization import initialize_network
from jbag.models.utils import get_norm_op, get_non_linear_op


class ConvBlock(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if non_linear_kwargs is None:
            non_linear_kwargs = {}

        self.activation = non_linear(**non_linear_kwargs)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = norm_op(mid_channels, **norm_op_kwargs)
        self.conv2 = nn.Conv2d(mid_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        output = self.activation(x)

        return output


class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels=1, num_classes=1,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 ):
        super(UNetPlusPlus, self).__init__()
        self.output_channels = num_classes

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(input_channels, filters[0], filters[0],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv1_0 = ConvBlock(filters[0], filters[1], filters[1],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv2_0 = ConvBlock(filters[1], filters[2], filters[2],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv3_0 = ConvBlock(filters[2], filters[3], filters[3],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv4_0 = ConvBlock(filters[3], filters[4], filters[4],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)

        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0], filters[0],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1], filters[1],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2], filters[2],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3], filters[3],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0], filters[0],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1], filters[1],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2], filters[2],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0], filters[0],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1], filters[1],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0], filters[0],
                                 norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                 non_linear=non_linear, non_linear_kwargs=non_linear_kwargs)

        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


def build_unet_plus_plus(network_config: Config):
    norm_op = get_norm_op(network_config.norm_op, network_config.conv_dim)
    non_linear_op = get_non_linear_op(network_config.non_linear)

    params = {'input_channels': network_config.input_channels,
              'num_classes': network_config.num_classes,
              'norm_op': norm_op,
              'norm_op_kwargs': network_config.norm_op_kwargs,
              'non_linear': non_linear_op,
              'non_linear_kwargs': network_config.non_linear_kwargs,
              }
    network = UNetPlusPlus(**params)

    initialize_network(network, network_config)
    return network
