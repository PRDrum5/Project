# Attempt to replicate the DCGAN model on the CIFAR dataset.
# This will provide a foundation on which to build more complex GAN models.

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import utils
import models
import loss_functions

# Use GPU if available
device = utils.use_gpu()

# Fix Random Seed
utils.fix_seed()

# ==============================================================================
# =                                Parameters                                  =
# ==============================================================================

logging = 100
model_name = './DCGAN'
batch_size = 128
num_epochs = 100

# Make file system for model and data
data_dir = '.././datasets'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_name):
    os.makedirs(model_name)

# ==============================================================================
# =                            Data Pre-Processing                             =
# ==============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

# Download and Construct Dataset
cifar10_train = datasets.CIFAR10(data_dir, 
                                 train=True, 
                                 download=True,
                                 transform=transform)

cifar10_val = datasets.CIFAR10(data_dir, 
                               train=True, 
                               download=True,
                               transform=transform)

cifar10_test = datasets.CIFAR10(data_dir, 
                                train=False, 
                                download=True, 
                                transform=transform)

train_size = 49000

loader_train = DataLoader(cifar10_train, 
                          batch_size=batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(train_size)))

loader_val = DataLoader(cifar10_val, 
                        batch_size=batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(train_size, 50000)))

loader_test = DataLoader(cifar10_test, batch_size=batch_size)

it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]
save_image(utils.denorm(fixed_input), os.path.join(model_name, 'input_sample.png'))

# ==============================================================================
# =                            Instantiate Models                              =
# ==============================================================================

weight_norm = 'weight_norm'
function_norm = 'batch_norm'
loss_mode = 'gan'
d_learning_rate  = 0.0002
g_learning_rate  = 0.0002
latent_z = 100

model_G = models.DC_Generator(z_dim=latent_z, 
                              norm_weights=weight_norm, 
                              norm_function=function_norm).to(device)
model_G.apply(utils.weights_init)
params_G = utils.param_count(model_G)

model_D = models.DC_Discriminator(norm_weights=weight_norm,
                                  norm_function=function_norm).to(device)
model_D.apply(utils.weights_init)
params_D = utils.param_count(model_D)

print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')
print("Total number of parameters is: {}".format(params_G + params_D))

# GAN loss function
d_loss_fn, g_loss_fn = loss_functions.get_losses_fn(loss_mode)

# Optimizer
d_optimizer = torch.optim.Adam(model_D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(model_G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))


# ==============================================================================
# =                              Train Model                                   =
# ==============================================================================
disc_training_losses = []
d_training_loss = 0

gen_training_losses = []
g_training_loss = 0

for epoch in range(num_epochs):
    for i, x in enumerate(loader_train, 0):

        # Enter training mode
        model_D.train()
        model_G.train()

        # Train Discriminator
        x = x.to(device)
        z = torch.randn(batch_size, latent_z).to(device)

        x_gen = model_G(z).detach()

        d_x_loss, d_x_gen_loss = d_loss_fn(model_D(x), model_D(x_gen))

        d_loss = d_x_loss + d_x_gen_loss
        d_training_loss += d_loss.item()

        model_D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_z).to(device)

        x_gen = model_G(z)

        g_loss = g_loss_fn(model_D(x_gen))
        g_training_loss += g_loss.item()

        model_G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % logging == 0:
            print("{:>6}/{:6} {:>6}/{:6} Loss D: {:f} Loss G: {:f}".format(
                epoch, num_epochs, i, len(loader_train), d_loss.item(), g_loss.item()))
    
    disc_training_losses.append(d_training_loss / len(loader_train))
    gen_training_losses.append(g_training_loss / len(loader_train))

# save losses and models
np.save(os.path.join(model_name, 'train_losses_G.npy'), np.array(gen_training_losses))
np.save(os.path.join(model_name, 'train_losses_D.npy'), np.array(disc_training_losses))
torch.save(model_G.state_dict(), os.path.join(model_name, 'DCGAN_model_G.pth'))
torch.save(model_D.state_dict(), os.path.join(model_name, 'DCGAN_model_D.pth'))
