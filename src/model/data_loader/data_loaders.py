import os
from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets import MelSpecBlendshapesDataset, SpecShapesToTensor

class DataLoaderMNIST(BaseDataLoader):
    """
    MNIST Dataloader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, 
                 train_split=0, n_workers=1, train=True, drop_last=False):

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.dataset = datasets.MNIST(self.data_dir, train=train, 
                                      download=True, transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, train_split, 
                         n_workers, drop_last=drop_last)

class DataLoaderCIFAR10(BaseDataLoader):
    """
    CIFAR10 Dataloader
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, train_split=0, n_workers=1, train=True):

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.CIFAR10(self.data_dir, train=True,
                                        download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, train_split, 
                         n_workers) 
                        
class DataLoaderMelSpecShapes(BaseDataLoader):
    """
    DataLoader for Mel Spectograms and Corresponding Blendshape Parameters
    """
    def __init__(self, melspec_dir, blendshapes_dir, batch_size,
                 shuffle=False, train_split=1, n_workers=1):

        self.melspec_dir = melspec_dir
        self.blendshapes_dir = blendshapes_dir

        transform = SpecShapesToTensor()

        self.dataset = MelSpecBlendshapesDataset(self.melspec_dir,
                                                 self.blendshapes_dir,
                                                 transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, 
                         train_split, n_workers)

