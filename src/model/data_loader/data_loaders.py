import os
from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets import *

class DataLoaderMFCCShapes(BaseDataLoader):
    """
    DataLoader for MFCC and Corresponding Blendshape Parameters
    """
    def __init__(self, melspec_dir, blendshapes_dir, batch_size,
                 shuffle=False, train_split=1, n_workers=1, drop_last=True):

        self.melspec_dir = melspec_dir
        self.blendshapes_dir = blendshapes_dir

        transform = SpecShapesToTensor()

        self.dataset = MFCCBlendshapesDataset(self.melspec_dir,
                                                 self.blendshapes_dir,
                                                 transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, 
                         train_split, n_workers, drop_last=drop_last)

class DataLoaderWavShapes(BaseDataLoader):
    """
    """
    def __init__(self, wav_path, blendshapes_path, batch_size, 
                 shuffle=False, n_workers=1, 
                 drop_last=True, tsfm=None):

        self.wav_path = wav_path
        self.blendshapes_path = blendshapes_path

        transform_list = []
        for t in tsfm:
            item = getattr(datasets, t)()
            transform_list.append(item)

        transform = transforms.Compose(transform_list)

        self.dataset = WavBlendshapesDataset(self.wav_path, 
                                             self.blendshapes_path,
                                             transform=transform)

        super().__init__(self.dataset, batch_size, shuffle,
                         n_workers, drop_last=drop_last)

class DataLoaderLrwShapes(BaseDataLoader):
    """
    LRW Shape parameters DataLoader
    """
    def __init__(self, blendshapes_dir, batch_size, shuffle=True,
                 n_workers=0, drop_last=True, tsfm=None):
                
        self.blendshapes_dir = blendshapes_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_workers = n_workers
        self.drop_last = drop_last

        transform_list = []
        for t in tsfm:
            item = getattr(datasets, t)()
            transform_list.append(item)
        
        transform = transforms.Compose(transform_list)

        self.dataset = LrwBlendshapesDataset(self.blendshapes_dir,
                                             transform=transform)

        super().__init__(self.dataset, self.batch_size, self.shuffle,
                         self.n_workers, drop_last=self.drop_last)
