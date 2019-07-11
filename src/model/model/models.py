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

class Voca_Shape_Generator(BaseModel):
    def __init__(self, z_dim=10, n_labels=1):
        super().__init__()

        in_dim = z_dim + n_labels

        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(4,1), 
                               stride=(1,1), bias=False)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,1), 
                               stride=(1,1), bias=False)
        self.relu2 = nn.ReLU()
    
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(4,2), 
                               stride=(2,1), bias=False)
        self.relu3 = nn.ReLU()
    
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(4,1), 
                               stride=(1,1), bias=False)
        self.relu4 = nn.ReLU()
    
        self.conv5 = nn.Conv2d(64, 32, kernel_size=(4,2), 
                               stride=(1,1), bias=False)
        self.relu5 = nn.ReLU()
    
        self.conv6 = nn.Conv2d(32, 16, kernel_size=(4,2), 
                               stride=(1,1), bias=False)
        self.relu6 = nn.ReLU()
    
        self.conv7 = nn.Conv2d(16, 6, kernel_size=(3,2), 
                               stride=(1,1), bias=False)
    
    def forward(self, noise, melspec):
        x = torch.cat((noise, melspec), dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        out = self.conv7(x)
        out = out.reshape(out.size(0), out.size(3), out.size(2), out.size(1))
        return out

class Voca_Shape_Critic(BaseModel):
    def __init__(self, spec_channels, shape_channels):
        super().__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, 4, 
                                    kernel_size=(4,1), bias=False)
        self.spec_relu1 = nn.ReLU()

        self.spec_conv2 = nn.Conv2d(4, 8, kernel_size=(4,1), bias=False)
        self.spec_relu2 = nn.ReLU()
    
        self.spec_conv3 = nn.Conv2d(8, 16, kernel_size=(4,2), 
                                    stride=(2,1), bias=False)
        self.spec_relu3 = nn.ReLU()
    
        self.spec_conv4 = nn.Conv2d(16, 32, kernel_size=(4,1), bias=False)
        self.spec_relu4 = nn.ReLU()
    
        self.spec_conv5 = nn.Conv2d(32, 64, kernel_size=(4,1), bias=False)
        self.spec_relu5 = nn.ReLU()
    
        self.spec_conv6 = nn.Conv2d(64, 128, kernel_size=(4,2), bias=False)
        self.spec_relu6 = nn.ReLU()
    
        self.spec_conv7 = nn.Conv2d(128, 128, kernel_size=(3,1), bias=False)
        self.spec_relu7 = nn.ReLU()

        self.shape_conv1 = nn.Conv2d(shape_channels, 16, 
                                     kernel_size=(1,2), bias=False)
        self.shape_relu1 = nn.ReLU()

        self.shape_conv2 = nn.Conv2d(16, 32, kernel_size=(1,2), bias=False)
        self.shape_relu2 = nn.ReLU()

        self.shape_conv3 = nn.Conv2d(32, 64, kernel_size=(1,2), bias=False)
        self.shape_relu3 = nn.ReLU()

        self.conv1 = nn.Conv2d(192, 128, kernel_size=(3,1), bias=False)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3,1), bias=False)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3,1), bias=False)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 16, kernel_size=(3,2), bias=False)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(16, 1, kernel_size=(2,2), bias=False)
        self.linear5 = nn.Linear(1,1)
    
    def forward(self, shapes, melspec):
        x = self.spec_relu1(self.spec_conv1(melspec))
        x = self.spec_relu2(self.spec_conv2(x))
        x = self.spec_relu3(self.spec_conv3(x))
        x = self.spec_relu4(self.spec_conv4(x))
        x = self.spec_relu5(self.spec_conv5(x))
        x = self.spec_relu6(self.spec_conv6(x))
        x = self.spec_relu7(self.spec_conv7(x))

        y = self.shape_relu1(self.shape_conv1(shapes))
        y = self.shape_relu2(self.shape_conv2(y))
        y = self.shape_relu3(self.shape_conv3(y))

        z = torch.cat((x, y), dim=1)

        z = self.relu1(self.conv1(z))
        z = self.relu2(self.conv2(z))
        z = self.relu3(self.conv3(z))
        z = self.relu4(self.conv4(z))
        out = self.linear5(self.conv5(z))

        return out

class MFCC_Shape_Gen(BaseModel):
    def __init__(self, z_dim=100, n_mfcc=50):
        super().__init__()

        in_dim = z_dim + n_mfcc

        self.tcon1 = nn.ConvTranspose1d(in_dim, 125, kernel_size=10, dilation=2)
        self.relu1 = nn.ReLU()

        self.tcon2 = nn.ConvTranspose1d(125, 100, kernel_size=9, dilation=2)
        self.relu2 = nn.ReLU()

        self.tcon3 = nn.ConvTranspose1d(100, 80, kernel_size=8, dilation=2)
        self.relu3 = nn.ReLU()

        self.tcon4 = nn.ConvTranspose1d(80, 60, kernel_size=7, dilation=2)
        self.relu4 = nn.ReLU()

        self.tcon5 = nn.ConvTranspose1d(60, 40, kernel_size=6, dilation=2)
        self.relu5 = nn.ReLU()

        self.tcon6 = nn.ConvTranspose1d(40, 30, kernel_size=5, dilation=2)
        self.relu6 = nn.ReLU()

        self.tcon7 = nn.ConvTranspose1d(30, 15, kernel_size=5, dilation=1)
        self.relu7 = nn.ReLU()

        self.tcon8 = nn.ConvTranspose1d(15, 10, kernel_size=4, dilation=1)

    def forward(self, noise, mfcc):
        x = torch.cat((noise, mfcc), dim=1)
        x = self.relu1(self.tcon1(x))
        x = self.relu2(self.tcon2(x))
        x = self.relu3(self.tcon3(x))
        x = self.relu4(self.tcon4(x))
        x = self.relu5(self.tcon5(x))
        x = self.relu6(self.tcon6(x))
        x = self.relu7(self.tcon7(x))
        x = self.tcon8(x)
        return x

class MFCC_Shape_Critic(BaseModel):
    def __init__(self, n_mfcc, n_shapes):
        super().__init__()

        # Process mfcc input
        self.mfcc_con1 = nn.Conv1d(n_mfcc, 128, kernel_size=10, dilation=2)
        self.mfcc_lrelu1 = nn.LeakyReLU(0.2)

        self.mfcc_con2 = nn.Conv1d(128, 128, kernel_size=9, dilation=2)
        self.mfcc_lrelu2 = nn.LeakyReLU(0.2)

        self.mfcc_con3 = nn.Conv1d(128, 128, kernel_size=8, dilation=2)
        self.mfcc_lrelu3 = nn.LeakyReLU(0.2)

        self.mfcc_con4 = nn.Conv1d(128, 64, kernel_size=8, dilation=2)
        self.mfcc_lrelu4 = nn.LeakyReLU(0.2)

        self.mfcc_con5 = nn.Conv1d(64, 64, kernel_size=8, dilation=2)
        self.mfcc_lrelu5 = nn.LeakyReLU(0.2)


        # Process blendshape parameter inputs
        self.shape_con1 = nn.Conv1d(n_shapes, 64, kernel_size=10, stride=2)
        self.shape_lrelu1 = nn.LeakyReLU(0.2)

        self.shape_con2 = nn.Conv1d(64, 64, kernel_size=8)
        self.shape_lrelu2 = nn.LeakyReLU(0.2)


        # Concat output of mfcc and shape networks
        # Process concat
        self.cat_con1 = nn.Conv1d(128, 128, kernel_size=7, stride=2)
        self.cat_lrelu1 = nn.LeakyReLU(0.2)

        self.cat_con2 = nn.Conv1d(128, 128, kernel_size=7, stride=2)
        self.cat_lrelu2 = nn.LeakyReLU(0.2)

        self.cat_con3 = nn.Conv1d(128, 128, kernel_size=5, stride=2)
        self.cat_lrelu3 = nn.LeakyReLU(0.2)

        self.cat_con4 = nn.Conv1d(128, 64, kernel_size=4)
        self.cat_lrelu4 = nn.LeakyReLU(0.2)

        self.cat_con5 = nn.Conv1d(64, 32, kernel_size=4)
        self.cat_lrelu5 = nn.LeakyReLU(0.2)

        self.cat_con6 = nn.Conv1d(32, 16, kernel_size=4)
        self.cat_lrelu6 = nn.LeakyReLU(0.2)

        self.cat_con7 = nn.Conv1d(16, 1, kernel_size=5)
        self.cat_lin7 = nn.Linear(1, 1)
    
    def forward(self, shapes, mfcc):

        x = self.mfcc_lrelu1(self.mfcc_con1(mfcc))
        x = self.mfcc_lrelu2(self.mfcc_con2(x))
        x = self.mfcc_lrelu3(self.mfcc_con3(x))
        x = self.mfcc_lrelu4(self.mfcc_con4(x))
        x = self.mfcc_lrelu5(self.mfcc_con5(x))

        y = self.shape_lrelu1(self.shape_con1(shapes))
        y = self.shape_lrelu2(self.shape_con2(y))

        z = torch.cat((x, y), dim=1)

        z = self.cat_lrelu1(self.cat_con1(z))
        z = self.cat_lrelu2(self.cat_con2(z))
        z = self.cat_lrelu3(self.cat_con3(z))
        z = self.cat_lrelu4(self.cat_con4(z))
        z = self.cat_lrelu5(self.cat_con5(z))
        z = self.cat_lrelu6(self.cat_con6(z))
        z = self.cat_lin7(self.cat_con7(z))
        return z





