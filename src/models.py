# Collection of models

import os
import torch
import torch.nn as nn
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image


class DC_Generator(nn.Module):
    def __init__(self, z_dim, dropout=0.5, norm_weights=None, norm_function=None):
        super(DC_Generator, self).__init__()

        self.dropout = dropout

        norm_fn = utils._get_norm_func_2D(norm_function)
        weight_norm_fn = utils._get_weight_norm_func(norm_weights)

        def conv_t_bn_relu_dp(in_dim, out_dim, kernel_size=3, stride=1, padding=0):
            """
            Transpose Convolution layer with weight normalization
            Batch norm applied
            ReLU 
            Dropout
            """
            layer = nn.Sequential(
                        weight_norm_fn(
                            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding)),
                        norm_fn(out_dim),
                        nn.ReLU())
            return layer

        self.gen_model = nn.Sequential(
            conv_t_bn_relu_dp(in_dim=z_dim, out_dim=256, kernel_size=3, stride=1),
            conv_t_bn_relu_dp(in_dim=256,   out_dim=256, kernel_size=3, stride=2),
            conv_t_bn_relu_dp(in_dim=256,   out_dim=128, kernel_size=3, stride=2),
            conv_t_bn_relu_dp(in_dim=128,   out_dim=64,  kernel_size=2, stride=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Carries out a forward pass through the generator model
        """
        out = self.gen_model(x)
        return out


class DC_Discriminator(nn.Module):
    def __init__(self, leak=0.2, dropout=0.5, norm_weights=None, norm_function=None):
        super(DC_Discriminator, self).__init__()

        self.leak = leak
        self.dropout = dropout

        norm_fn = utils._get_norm_func_2D(norm_function)
        weight_norm_fn = utils._get_weight_norm_func(norm_weights)

        def conv_bn_lrelu_dp(in_dim, out_dim, kernel_size=3, stride=1, padding=0):
            """
            Convolution layer with weight normalization
            Batch norm applied
            Leaky ReLU 
            Dropout
            """
            layer = nn.Sequential(
                        weight_norm_fn(
                            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                        norm_fn(out_dim),
                        nn.LeakyReLU(self.leak),
                        nn.Dropout(self.dropout),
                        )
            return layer

        self.disc_model = nn.Sequential(
            conv_bn_lrelu_dp(in_dim=3,   out_dim=128, kernel_size=3, stride=1),
            conv_bn_lrelu_dp(in_dim=128, out_dim=128, kernel_size=3, stride=1),
            conv_bn_lrelu_dp(in_dim=128, out_dim=256, kernel_size=3, stride=1),
            conv_bn_lrelu_dp(in_dim=256, out_dim=256, kernel_size=2, stride=2),
            conv_bn_lrelu_dp(in_dim=256, out_dim=128, kernel_size=2, stride=2),
            conv_bn_lrelu_dp(in_dim=128, out_dim=128, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Carries out a forward pass through the discriminator model
        """
        out = self.disc_model(x)
        return out.view(-1, 1).squeeze(1)
