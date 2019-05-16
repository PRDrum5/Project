# GAN module

import torch.nn as nn

class NoOp(nn.Module):

    def __init__(self, *args, **keyword_args):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


def identity(x, *args, **keyword_args):
    return x


def _get_norm_func_2D(norm):  # 2d
    if norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == None:
        return NoOp
    else:
        raise NotImplementedError


def _get_weight_norm_func(weight_norm):
    if weight_norm == 'spectral_norm':
        return nn.utils.spectral_norm
    elif weight_norm == 'weight_norm':
        return nn.utils.weight_norm
    elif weight_norm == None:
        return identity
    else:
        return NotImplementedError


class Generator(nn.Module):
    def __init__(self, norm_weights=None, norm_function=None):
        super(Generator, self).__init__()

        self.norm_fn = _get_norm_func_2D(norm_function)
        self.weight_norm_fn = _get_weight_norm_func(norm_weights)

        def forward(self):
            pass

class Discriminator(nn.Module):
    def __init__(self, norm_weights=None, norm_function=None):
        super(Discriminator, self).__init__()

        self.norm_fn = _get_norm_func_2D(norm_function)
        self.weight_norm_fn = _get_weight_norm_func(norm_weights)

        def forward(self):
            pass
