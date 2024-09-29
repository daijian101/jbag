from typing import Union, Type

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from jbag.config import Config
from jbag.models.network_weight_initialization import initialize_network
from jbag.models.utils import get_conv_op, get_norm_op, get_non_linear_op, get_matching_conv_transpose_op, \
    get_conv_dimensions


class UNet(nn.Module):
    def __init__(self, input_channels: int,
                 num_classes: int,
                 num_stages: int,
                 num_features_per_stage: Union[int, list[int], tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, list[int], tuple[int, ...]],
                 strides: Union[int, list[int], tuple[int, ...]],
                 num_conv_per_stage_encoder: Union[int, list[int], tuple[int, ...]],
                 num_conv_per_stage_decoder: Union[int, list[int], tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 deep_supervision: bool = False,
                 non_linear_first: bool = False
                 ):
        super().__init__()

        if isinstance(num_conv_per_stage_encoder, int):
            num_conv_per_stage_encoder = [num_conv_per_stage_encoder] * num_stages

        if isinstance(num_conv_per_stage_decoder, int):
            num_conv_per_stage_decoder = [num_conv_per_stage_decoder] * (num_stages - 1)

        assert len(num_conv_per_stage_encoder) == num_stages
        assert len(num_conv_per_stage_decoder) == num_stages - 1
        self.encoder = Encoder(input_channels=input_channels,
                               num_stages=num_stages,
                               num_features_per_stage=num_features_per_stage,
                               conv_op=conv_op,
                               kernel_sizes=kernel_sizes,
                               strides=strides,
                               num_conv_per_stage=num_conv_per_stage_encoder,
                               conv_bias=conv_bias,
                               norm_op=norm_op,
                               norm_op_kwargs=norm_op_kwargs,
                               dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs,
                               non_linear=non_linear,
                               non_linear_kwargs=non_linear_kwargs,
                               return_skips=True,
                               non_linear_first=non_linear_first
                               )
        self.decoder = Decoder(num_classes=num_classes,
                               encoder=self.encoder,
                               num_conv_per_stage=num_conv_per_stage_decoder,
                               deep_supervision=deep_supervision,
                               non_linear_first=non_linear_first
                               )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


class Encoder(nn.Module):
    def __init__(self, input_channels: int,
                 num_stages: int,
                 num_features_per_stage: Union[int, list[int], tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, list[int], tuple[int, ...]],
                 strides: Union[int, list[int], tuple[int, ...]],
                 num_conv_per_stage: Union[int, list[int], tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 return_skips: bool = False,
                 non_linear_first: bool = False
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_stages
        if isinstance(num_features_per_stage, int):
            num_features_per_stage = [num_features_per_stage] * num_stages
        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = [num_conv_per_stage] * num_stages
        if isinstance(strides, int):
            strides = [strides] * num_stages

        assert len(kernel_sizes) == num_stages
        assert len(num_features_per_stage) == num_stages
        assert len(num_conv_per_stage) == num_stages
        assert len(strides) == num_stages

        stages = []
        for i in range(num_stages):
            conv_stride = strides[i]
            stages.append(StackedConvBlock(input_channels=input_channels,
                                           output_channels=num_features_per_stage[i],
                                           num_convs=num_conv_per_stage[i],
                                           conv_op=conv_op,
                                           kernel_size=kernel_sizes[i],
                                           initial_stride=conv_stride,
                                           conv_bias=conv_bias,
                                           norm_op=norm_op,
                                           norm_op_kwargs=norm_op_kwargs,
                                           dropout_op=dropout_op,
                                           dropout_op_kwargs=dropout_op_kwargs,
                                           non_linear=non_linear,
                                           non_linear_kwargs=non_linear_kwargs,
                                           non_linear_first=non_linear_first
                                           ))
            input_channels = num_features_per_stage[i]

        self.stages = nn.ModuleList(stages)
        self.return_skips = return_skips
        self.output_channels = num_features_per_stage
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.non_linear = non_linear
        self.non_linear_kwargs = non_linear_kwargs

    def forward(self, x):
        ret = []
        for layer in self.stages:
            x = layer(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class Decoder(nn.Module):
    def __init__(self, num_classes: int,
                 encoder: Encoder,
                 num_conv_per_stage: Union[int, list[int], tuple[int, ...]],
                 deep_supervision: bool,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 conv_bias: bool = None,
                 non_linear_first: bool = False
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        num_stages_encoder = len(encoder.output_channels)
        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = [num_conv_per_stage] * (num_stages_encoder - 1)
        assert len(num_conv_per_stage) == num_stages_encoder - 1
        conv_transpose_op = get_matching_conv_transpose_op(encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        non_linear = encoder.non_linear if non_linear is None else non_linear
        non_linear_kwargs = encoder.non_linear_kwargs if non_linear_kwargs is None else non_linear_kwargs

        stages = []
        conv_transpose_ops = []
        seg_layers = []
        for i in range(1, num_stages_encoder):
            input_features_below = encoder.output_channels[-i]
            input_features_skip = encoder.output_channels[-(i + 1)]
            stride_for_transpose_conv = encoder.strides[-i]
            conv_transpose_ops.append(conv_transpose_op(input_features_below,
                                                        input_features_skip,
                                                        stride_for_transpose_conv,
                                                        stride_for_transpose_conv,
                                                        bias=conv_bias
                                                        ))

            stages.append(StackedConvBlock(
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                num_convs=num_conv_per_stage[i - 1],
                conv_op=encoder.conv_op,
                kernel_size=encoder.kernel_sizes[-(i + 1)],
                initial_stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                non_linear=non_linear,
                non_linear_kwargs=non_linear_kwargs,
                non_linear_first=non_linear_first
            ))

            seg_layers.append(
                encoder.conv_op(input_features_skip, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.conv_transpose_ops = nn.ModuleList(conv_transpose_ops)

        # Deep supervision is added anywhere, even for deep_supervision = False.
        # This allows load pre-trained parameters with and without deep supervision simultaneously.
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        low_resolution_features = skips[-1]
        seg_outputs = []
        for i in range(len(self.stages)):
            x = self.conv_transpose_ops[i](low_resolution_features)
            x = torch.cat((x, skips[-(i + 2)]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[i](x))
            elif i == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            low_resolution_features = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs


class StackedConvBlock(nn.Sequential):
    def __init__(self, input_channels: int,
                 output_channels: int,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 kernel_size: Union[int, list[int], tuple[int, ...]],
                 initial_stride: Union[int, list[int], tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 non_linear_first: bool = False
                 ):
        if not isinstance(output_channels, (list, tuple)):
            output_channels = [output_channels] * num_convs

        super().__init__(ConvDropoutNormReLU(input_channels=input_channels,
                                             output_channels=output_channels[0],
                                             conv_op=conv_op,
                                             kernel_size=kernel_size,
                                             stride=initial_stride,
                                             conv_bias=conv_bias,
                                             norm_op=norm_op,
                                             norm_op_kwargs=norm_op_kwargs,
                                             dropout_op=dropout_op,
                                             dropout_op_kwargs=dropout_op_kwargs,
                                             non_linear=non_linear,
                                             non_linear_kwargs=non_linear_kwargs,
                                             non_linear_first=non_linear_first),
                         *[
                             ConvDropoutNormReLU(input_channels=output_channels[i - 1],
                                                 output_channels=output_channels[i],
                                                 conv_op=conv_op,
                                                 kernel_size=kernel_size,
                                                 stride=1,
                                                 conv_bias=conv_bias,
                                                 norm_op=norm_op,
                                                 norm_op_kwargs=norm_op_kwargs,
                                                 dropout_op=dropout_op,
                                                 dropout_op_kwargs=dropout_op_kwargs,
                                                 non_linear=non_linear,
                                                 non_linear_kwargs=non_linear_kwargs,
                                                 non_linear_first=non_linear_first)
                             for i in range(1, num_convs)
                         ]
                         )


class ConvDropoutNormReLU(nn.Sequential):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 conv_op: Type[_ConvNd],
                 kernel_size: Union[int, list[int], tuple[int, ...]],
                 stride: Union[int, list[int], tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 non_linear_first: bool = False
                 ):

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if non_linear_kwargs is None:
            non_linear_kwargs = {}

        ops = []

        conv_dim = get_conv_dimensions(conv_op)
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * conv_dim

        conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias
        )

        ops.append(conv)

        if dropout_op is not None:
            dropout = dropout_op(**dropout_op_kwargs)
            ops.append(dropout)

        if norm_op is not None:
            norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(norm)

        if non_linear is not None:
            non_linear = non_linear(**non_linear_kwargs)
            ops.append(non_linear)

        if non_linear_first and (norm_op is not None and non_linear is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        super().__init__(*ops)


def build_unet(network_config: Config):
    conv_op = get_conv_op(network_config.conv_dim)
    norm_op = get_norm_op(network_config.norm_op, network_config.conv_dim)
    non_linear_op = get_non_linear_op(network_config.non_linear)

    params = {'input_channels': network_config.input_channels,
              'num_classes': network_config.num_classes,
              'num_stages': network_config.num_stages,
              'num_features_per_stage': network_config.num_features_per_stage,
              'conv_op': conv_op,
              'kernel_sizes': network_config.kernel_sizes,
              'strides': network_config.strides,
              'num_conv_per_stage_encoder': network_config.num_conv_per_stage_encoder,
              'num_conv_per_stage_decoder': network_config.num_conv_per_stage_decoder,
              'conv_bias': network_config.conv_bias,
              'norm_op': norm_op,
              'norm_op_kwargs': network_config.norm_op_kwargs,
              'non_linear': non_linear_op,
              'non_linear_kwargs': network_config.non_linear_kwargs,
              'deep_supervision': network_config.deep_supervision,
              'non_linear_first': network_config.non_linear_first,
              }
    network = UNet(**params)

    initialize_network(network, network_config)

    return network
