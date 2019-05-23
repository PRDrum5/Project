# Conditional GANS model with MNIST dataset

import torch
import torch.nn as nn
from gan_template import Generator, Discriminator

class Cond_Generator(Generator):
    def __init__(self, c_dim, z_dim, norm_weights=None, norm_function=None):
        super(Cond_Generator, self). __init__(norm_weights=norm_weights,
                                              norm_function=norm_function)
        
        def conv_t_bn_relu(in_dim, out_dim, kernel_size=3, stride=1, padding=0):
            """
            Transpose Convolution layer with weight norm
            batch norm applied
            ReLu
            """
            layer = nn.Sequential(
                        self.weight_norm_fn(
                            nn.ConvTranspose2d(in_dim, 
                                               out_dim, 
                                               kernel_size, 
                                               stride, 
                                               padding)),
                        self.norm_fn(out_dim),
                        nn.ReLU())
            return layer
        
        self.gen_model = nn.Sequential(
            conv_t_bn_relu(in_dim=c_dim+z_dim, out_dim=512, kernel_size=4, stride=1),
            conv_t_bn_relu(in_dim=512,         out_dim=256, kernel_size=3, stride=2, padding=1),
            conv_t_bn_relu(in_dim=256,         out_dim=128, kernel_size=3, stride=2, padding=1),
            conv_t_bn_relu(in_dim=128,         out_dim=64 , kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4),
            nn.Tanh()
        )

        def forward(self, c, z):
            # z: (N, z_dim), c: (N, c_dim)
            x = torch.cat([z, c], 1)
            x = self.gen_model(x.view(x.size(0), x.size(1), 1, 1))
            return x
        