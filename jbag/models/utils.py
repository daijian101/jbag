import torch.nn as nn


def get_conv_op(dim):
    match dim:
        case 1:
            return nn.Conv1d
        case 2:
            return nn.Conv2d
        case 3:
            return nn.Conv3d
        case _:
            raise ValueError()


def get_norm_op(op_name, dim):
    match op_name:
        case 'InstanceNorm':
            match dim:
                case 2:
                    return nn.InstanceNorm2d
        case 'BatchNorm':
            match dim:
                case 2:
                    return nn.BatchNorm2d


def get_non_linear_op(op_name):
    match op_name:
        case 'leaky_relu':
            return nn.LeakyReLU
        case 'relu':
            return nn.ReLU
