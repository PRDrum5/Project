import torch
from torch.utils.data import Dataset
import os

class MelSpecBlendshapesDataset(Dataset):
    """
    Mel Spectogram features
    Blendshape Parameters targets
    """

    def __init__(self, melspec_dir, blendshapes_dir, transform=None):
        self.melspec_dir = melspec_dir
        self.blendshapes_dir = blendshapes_dir
        self.transform = transform

    def __len__(self):
        _path, _dirs, files = next(os.walk(self.melspec_dir))
        length = len(files)
        return length
    
    def __getitem__(self, idx):
        pass
        #TODO get name of npy files in blendshapes, use this to fetch spec file