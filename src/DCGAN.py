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
from utils import denorm

# Use GPU if available
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


# Fix Random Seed
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


### Parameters ###
model_name = './DCGAN'
batch_size = 128
NUM_TRAIN = 49000
num_epochs = 100
learning_rate  = 0.0002
latent_vector_size = 100

data_dir = '.././datasets'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_name):
    os.makedirs(model_name)

# Data Pre-Processing
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

loader_train = DataLoader(cifar10_train, 
                          batch_size=batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

loader_val = DataLoader(cifar10_val, 
                        batch_size=batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

loader_test = DataLoader(cifar10_test, batch_size=batch_size)

it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]
save_image(denorm(fixed_input), os.path.join(model_name, 'input_sample.png'))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_t1 = nn.ConvTranspose2d(100, 512, kernel_size=2, stride=1, bias=False)
        self.conv_t2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.conv_t3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.conv_t4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.conv_t5 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=False)
        
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        
        self.tanh = nn.Tanh()
        

    def decode(self, x):
        x = self.conv_t1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv_t2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.conv_t3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        
        x = self.conv_t4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        
        x = self.conv_t5(x)
        x = self.tanh(x)

        return x

    def forward(self, z):
        print("z = ", z.shape)
        1/0
        return self.decode(z)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=2, stride=1, padding=0, bias=False)
        
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.sigmoid = nn.Sigmoid()
        
    def discriminator(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv5(x)
        out = self.sigmoid(x)
        
        return out

    def forward(self, x):
        print("x = ", x.shape)
        1/0
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


use_weights_init = True

model_G = Generator().to(device)
if use_weights_init:
    model_G.apply(weights_init)
params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')

model_D = Discriminator().to(device)
if use_weights_init:
    model_D.apply(weights_init)
params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')

print("Total number of parameters is: {}".format(params_G + params_D))

criterion = nn.BCELoss(reduction='mean')
def loss_function(out, label):
    loss = criterion(out, label)
    return loss

# setup optimizer
# You are free to add a scheduler or change the optimizer if you want. We chose one for you for simplicity.
beta1 = 0.5
optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))


fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
real_label = 1
fake_label = 0


export_folder = model_name
train_losses_G = []
train_losses_D = []

logging = 10

for epoch in range(num_epochs):
    for i, data in enumerate(loader_train, 0):
        train_loss_D = 0
        train_loss_G = 0
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        model_D.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = model_D(real_cpu)
        errD_real = loss_function(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
        fake = model_G(noise)
        label.fill_(fake_label)
        output = model_D(fake.detach())
        errD_fake = loss_function(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        train_loss_D += errD.item()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model_G.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = model_D(fake)
        errG = loss_function(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        train_loss_G += errG.item()
        optimizerG.step()
        
        if i % logging == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(loader_train),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    if epoch == 0:
        save_image(denorm(real_cpu.cpu()), os.path.join(model_name, 'real_samples.png'))
    
    fake = model_G(fixed_noise)
    save_image(denorm(fake.cpu()), os.path.join(model_name, 'fake_samples_epoch_%03d.png' % epoch))
    train_losses_D.append(train_loss_D / len(loader_train))
    train_losses_G.append(train_loss_G / len(loader_train))
            
# save losses and models
np.save(os.path.join(model_name, 'train_losses_G.npy', np.array(train_losses_G)))
np.save(os.path.join(model_name, 'train_losses_D.npy', np.array(train_losses_D)))
torch.save(model_G.state_dict(), os.path.join(model_name, 'DCGAN_model_G.pth'))
torch.save(model_D.state_dict(), os.path.join(model_name, 'DCGAN_model_D.pth'))
