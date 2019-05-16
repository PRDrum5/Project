# Conditional GANS model with MNIST dataset

import torch.nn as nn
from gan_template import Generator, Discriminator

class Cond_Generator(Generator):
    def __init__(self, c_dim, z_dim, norm_weights=None, norm_function=None):
        super(Cond_Generator, self). __init__(norm_weights=norm_weights,
                                              norm_function=norm_function)
        

        

