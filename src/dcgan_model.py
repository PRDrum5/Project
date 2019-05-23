# DCGANs model fro CIFAR10 

import torch.nn as nn
from gan_template import Generator, Discriminator


class Conv_Generator(Generator):
    def __init__(self, z_dim, norm_weights=None, norm_function=None):
        super(Conv_Generator, self).__init__(norm_weights=norm_weights, 
                                             norm_function=norm_function)

        def conv_t_bn_relu(in_dim, out_dim, kernel_size=3, stride=1, padding=0):
            """
            Transpose Convolution layer with weight normalization
            Batch norm applied
            ReLU 
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
            conv_t_bn_relu(in_dim=z_dim, out_dim=512, kernel_size=2, stride=1), # 1 -> 2
            conv_t_bn_relu(in_dim=512,   out_dim=256, kernel_size=2, stride=2), # 2 -> 4
            conv_t_bn_relu(in_dim=256,   out_dim=128, kernel_size=2, stride=2), # 4 -> 8
            conv_t_bn_relu(in_dim=128,   out_dim=64,  kernel_size=2, stride=2), # 8 -> 16
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2), # 16 -> 32
            nn.Tanh()
        )

    def forward(self, x):
        """
        Carries out a forward pass through the generator model
        """
        out = self.gen_model(x)
        return out


class Conv_Discriminator(Discriminator):
    def __init__(self, leak=0.2, dropout=0.5, norm_weights=None, norm_function=None):
        super(Conv_Discriminator, self).__init__(norm_weights=norm_weights, 
                                                 norm_function=norm_function)

        self.leak = leak
        self.dropout = dropout

        def conv_bn_lrelu_dp(in_dim, out_dim, kernel_size=3, stride=1, padding=0):
            """
            Convolution layer with weight normalization
            Batch norm applied
            Leaky ReLU 
            Dropout
            """
            layer = nn.Sequential(
                        self.weight_norm_fn(nn.Conv2d(in_dim, 
                                                      out_dim, 
                                                      kernel_size, 
                                                      stride, 
                                                      padding)),
                        self.norm_fn(out_dim),
                        nn.LeakyReLU(self.leak),
                        nn.Dropout(self.dropout),
                        )
            return layer

        self.disc_model = nn.Sequential(
            conv_bn_lrelu_dp(in_dim=3,   out_dim=64,  kernel_size=2, stride=1), # 32 -> 31
            conv_bn_lrelu_dp(in_dim=64,  out_dim=128, kernel_size=2, stride=1), # 31 -> 30
            conv_bn_lrelu_dp(in_dim=128, out_dim=256, kernel_size=2, stride=2), # 30 -> 15
            conv_bn_lrelu_dp(in_dim=256, out_dim=512, kernel_size=3, stride=2), # 15 -> 7
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=2),#  7 -> 3
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Carries out a forward pass through the discriminator model
        """
        out = self.disc_model(x)
        return out.view(-1, 1).squeeze(1)
