# Attempt to replicate the DCGAN model on the CIFAR dataset.
# This will provide a foundation on which to build more complex GAN models.

import os
import torch
import torch.nn as nn
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
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)