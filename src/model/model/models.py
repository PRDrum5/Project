import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Generator_MNIST(BaseModel):
    def __init__(self, z_dim=100):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(z_dim, 256, kernel_size=4, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.convt4 = nn.ConvTranspose2d(64, 64, kernel_size=4, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.convt5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.convt6 = nn.ConvTranspose2d(32, 16, kernel_size=3, bias=False)
        self.bn6 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()

        self.convt7 = nn.ConvTranspose2d(16, 1, kernel_size=3, bias=False)
        self.tanh7 = nn.Tanh()


    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.relu1(self.bn1(self.convt1(x)))
        x = self.relu2(self.bn2(self.convt2(x)))
        x = self.relu3(self.bn3(self.convt3(x)))
        x = self.relu4(self.bn4(self.convt4(x)))
        x = self.relu5(self.bn5(self.convt5(x)))
        x = self.relu6(self.bn6(self.convt6(x)))
        out = self.tanh7(self.convt7(x))
        return out

class Discriminator_MNIST(BaseModel):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, bias=False)
        self.leaky_relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.leaky_relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.leaky_relu3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout2d(0.2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.leaky_relu4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout2d(0.2)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.leaky_relu5 = nn.LeakyReLU()
        self.drop5 = nn.Dropout2d(0.2)

        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.leaky_relu6 = nn.LeakyReLU()
        self.drop6 = nn.Dropout2d(0.2)

        self.conv7 = nn.Conv2d(32, 1, kernel_size=4, bias=False)
        self.sigmoid7 = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.leaky_relu1(self.conv1(x)))
        x = self.drop2(self.leaky_relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.leaky_relu3(self.bn3(self.conv3(x))))
        x = self.drop4(self.leaky_relu4(self.bn4(self.conv4(x))))
        x = self.drop5(self.leaky_relu5(self.bn5(self.conv5(x))))
        x = self.drop6(self.leaky_relu6(self.bn6(self.conv6(x))))
        out = self.sigmoid7(self.conv7(x))
        out = torch.squeeze(out)
        return out

class Critic_MNIST(BaseModel):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, bias=False)
        self.leaky_relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        self.leaky_relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.leaky_relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, bias=False)
        self.leaky_relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.leaky_relu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(256, 64, kernel_size=3, bias=False)
        self.leaky_relu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(64, 1, kernel_size=4, bias=False)
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.leaky_relu5(self.conv5(x))
        x = self.leaky_relu6(self.conv6(x))
        out = self.linear(self.conv7(x))
        out = torch.squeeze(out)
        return out

class Cond_Generator_MNIST(BaseModel):
    def __init__(self, z_dim=100, n_labels=10):
        super().__init__()

        in_dim = z_dim + n_labels

        self.convt1 = nn.ConvTranspose2d(in_dim, 256, kernel_size=4, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.convt4 = nn.ConvTranspose2d(64, 64, kernel_size=4, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.convt5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.convt6 = nn.ConvTranspose2d(32, 16, kernel_size=3, bias=False)
        self.bn6 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()

        self.convt7 = nn.ConvTranspose2d(16, 1, kernel_size=3, bias=False)
        self.tanh7 = nn.Tanh()

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.relu1(self.bn1(self.convt1(x)))
        x = self.relu2(self.bn2(self.convt2(x)))
        x = self.relu3(self.bn3(self.convt3(x)))
        x = self.relu4(self.bn4(self.convt4(x)))
        x = self.relu5(self.bn5(self.convt5(x)))
        x = self.relu6(self.bn6(self.convt6(x)))
        out = self.tanh7(self.convt7(x))
        return out

class Cond_Critic_MNIST(BaseModel):
    def __init__(self, in_channels, n_labels):
        super().__init__()

        in_dim = in_channels + n_labels

        self.conv1 = nn.Conv2d(in_dim, 16, kernel_size=3, bias=False)
        self.leaky_relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        self.leaky_relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.leaky_relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, bias=False)
        self.leaky_relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.leaky_relu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(256, 64, kernel_size=3, bias=False)
        self.leaky_relu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(64, 1, kernel_size=4, bias=False)
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x, c):
        c = c.unsqueeze(-1).unsqueeze(-1)
        c = c.expand((c.size(0), c.size(1), x.size(2), x.size(3)))
        x = torch.cat((x, c), dim=1)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.leaky_relu5(self.conv5(x))
        x = self.leaky_relu6(self.conv6(x))
        out = self.linear(self.conv7(x))
        out = torch.squeeze(out)
        return out