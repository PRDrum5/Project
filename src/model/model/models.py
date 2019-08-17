import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

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

class Mfcc_Shape_Gen(BaseModel):
    def __init__(self, z_dim, shapes_dim):
        super().__init__()

        in_dim = z_dim + 1

        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=(4,3), 
                                           stride=(2,1), 
                                           padding=(0,7))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,3), stride=(2,1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1))
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1))
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(64, 4, kernel_size=(3,3), stride=(1,1))
        self.sig7 = nn.Sigmoid()


    def forward(self, noise, mfcc):
        x = torch.cat((noise, mfcc), dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.sig7(self.conv7(x))
        return x

class Mfcc_Shape_Gen_Shrink(BaseModel):
    def __init__(self, z_dim, shapes_dim):
        super().__init__()

        in_dim = z_dim + 1

        self.conv1 = nn.Conv2d(in_dim, 256, kernel_size=(4,3), 
                                           stride=(2,1), 
                                           padding=(0,7))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 256, kernel_size=(4,3), stride=(2,1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1))
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1))
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 4, kernel_size=(3,3), stride=(1,1))
        self.sig7 = nn.Sigmoid()


    def forward(self, noise, mfcc):
        x = torch.cat((noise, mfcc), dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.sig7(self.conv7(x))
        return x

class Mfcc_Shape_Gen_Small(BaseModel):
    def __init__(self, z_dim, shapes_dim):
        super().__init__()

        in_dim = z_dim + 1

        self.conv1 = nn.Conv2d(in_dim, 256, kernel_size=(4,3), 
                                           stride=(2,1), 
                                           padding=(0,5))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 128, kernel_size=(4,3), stride=(2,1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(2,1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 4, kernel_size=(3,3), stride=(1,1))
        self.sig5 = nn.Sigmoid()

    def forward(self, noise, mfcc):
        batch_size = mfcc.size(0)
        x = torch.cat((noise, mfcc), dim=1)
    
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.sig5(self.conv5(x))
        return x

class Mfcc_Shape_Gen_Small_2(BaseModel):
    def __init__(self, z_dim, shapes_dim):
        super().__init__()

        in_dim = z_dim + 1

        self.conv1 = nn.Conv2d(in_dim, 256, kernel_size=(4,3), 
                                           stride=(2,1), 
                                           padding=(0,6))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 128, kernel_size=(4,3), stride=(2,1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(2,1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 4, kernel_size=(3,5), stride=(1,1))
        self.sig5 = nn.Sigmoid()

    def forward(self, noise, mfcc):
        batch_size = mfcc.size(0)
        x = torch.cat((noise, mfcc), dim=1)
    
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.sig5(self.conv5(x))
        return x

class Mfcc_Shape_Gen_Small_Lin(BaseModel):
    def __init__(self, z_dim, shapes_dim):
        super().__init__()

        in_dim = z_dim + 1

        self.conv1 = nn.Conv2d(in_dim, 256, kernel_size=(4,3), 
                                           stride=(2,1), 
                                           padding=(0,6))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 128, kernel_size=(4,3), stride=(2,1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(2,1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 4, kernel_size=(3,3), stride=(1,1))
        self.relu5 = nn.ReLU()

        self.lin6 = nn.Linear(180, 172)
        self.sig6 = nn.Sigmoid()

    def forward(self, noise, mfcc):
        batch_size = mfcc.size(0)
        x = torch.cat((noise, mfcc), dim=1)
    
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))

        x = x.view(batch_size, 180)
        x = self.sig6(self.lin6(x))
        x = x.view(batch_size, 4, 1, 43)
        return x

class Mfcc_Shape_Critic(BaseModel):
    def __init__(self, shapes_dim):
        super().__init__()

        in_dim = shapes_dim + 1

        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(4,1), stride=(2,1))
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,1), stride=(2,1))
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,1), stride=(2,1))
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1))
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(2,1))
        self.lrelu5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv1d(256, 256, kernel_size=5, stride=2)
        self.lrelu6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv1d(256, 128, kernel_size=4, stride=2)
        self.lrelu7 = nn.LeakyReLU(0.2)

        self.conv8 = nn.Conv1d(128, 64, kernel_size=3, stride=2)
        self.lrelu8 = nn.LeakyReLU(0.2)

        self.lin9 = nn.Linear(256, 16)
        self.tanh9 = nn.Tanh()

    
    def forward(self, shapes, mfcc):
        batch_size = mfcc.size(0)

        # Expand shapes to same shape as mfcc
        height, width = mfcc.size(2), mfcc.size(3)
        ones = torch.ones(height, width, device=shapes.device)
        shapes = shapes * ones

        x = torch.cat((shapes, mfcc), dim=1)
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        x = self.lrelu5(self.conv5(x))
        x = x.squeeze(2)
        x = self.lrelu6(self.conv6(x))
        x = self.lrelu7(self.conv7(x))
        x = self.lrelu8(self.conv8(x))

        x = x.view(batch_size, -1)
        x = self.tanh9(self.lin9(x))
        return x

class Shape_Critic(BaseModel):
    def __init__(self, shapes_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(shapes_dim, 32, kernel_size=3)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv1d(256, 256, kernel_size=3)
        self.lrelu5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv1d(256, 256, kernel_size=3)
        self.lrelu6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv1d(256, 128, kernel_size=4, stride=2)
        self.lrelu7 = nn.LeakyReLU(0.2)

        self.conv8 = nn.Conv1d(128, 64, kernel_size=3)
        self.lrelu8 = nn.LeakyReLU(0.2)

        self.lin9 = nn.Linear(256, 16)
        self.tanh9 = nn.Tanh()

    def forward(self, shapes):
        batch_size = shapes.size(0)

        x = shapes.squeeze(2)

        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        x = self.lrelu5(self.conv5(x))
        x = self.lrelu6(self.conv6(x))
        x = self.lrelu7(self.conv7(x))
        x = self.lrelu8(self.conv8(x))

        x = x.view(batch_size, -1)
        x = self.tanh9(self.lin9(x))
        return x

class Mfcc_Multi_Towers_Classifier(BaseModel):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(3,1))
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.0)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=(2,1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.0)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.0)
        self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.0)
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        self.relu5 = nn.ReLU()

        self.lin6 = nn.Linear(448, 500)
        self.softmax6 = nn.LogSoftmax()

    def forward(self, shapes):
        """
        shapes: channel = n_params, length = n_frames
        shapes: [batch_size, 1, 4, 43]
        """
        batch_size = shapes.size(0)

        x = self.relu1(self.bn1(self.conv1(shapes)))
        x = self.drop1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.drop2(x)

        x = x.squeeze(2)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.max_pool3(x)

        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.drop4(x)
        x = self.max_pool4(x)

        x = self.relu5(self.conv5(x))

        x = x.view(batch_size, -1)
        x = self.softmax6(self.lin6(x))

        return x

class Lrw_Shape_Classifier(BaseModel):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(512, 256, kernel_size=3)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0)

        self.lin7 = nn.Linear(1280, 500)
        self.softmax7 = nn.LogSoftmax()

    
    def forward(self, shapes):
        """
        shapes: channel = n_params, length = n_frames
        shapes: [batch_size, 1, 4, 43]
        """
        batch_size = shapes.size(0)

        shapes = shapes.squeeze(1)

        x = self.relu1(self.bn1(self.conv1(shapes)))
        x = self.drop1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.max_pool3(x)

        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.drop4(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.max_pool5(x)

        x = self.relu6(self.conv6(x))
        x = self.drop6(x)
        x = x.view(batch_size, -1)
        x = self.softmax7(self.lin7(x))

        return x
