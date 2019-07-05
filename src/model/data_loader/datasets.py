import torch
from torch.utils.data import Dataset
import os
import numpy as np

class MelSpecBlendshapesDataset(Dataset):
    """
    Mel Spectogram features
    Blendshape Parameters targets
    """

    def __init__(self, melspec_dir, blendshapes_dir, transform=None):
        self.melspec_dir = melspec_dir
        self.blendshapes_dir = blendshapes_dir
        self.transform = transform
        self.melspec_list = sorted(os.listdir(melspec_dir))

    def __len__(self):
        _path, _dirs, files = next(os.walk(self.melspec_dir))
        length = len(files)
        return length
    
    def __getitem__(self, idx):
        item_name = self.melspec_list[idx]

        melspec = np.load(os.path.join(self.melspec_dir, item_name))
        shape_param = np.load(os.path.join(self.blendshapes_dir, item_name))

        sample = {'melspec': melspec, 'shape_param': shape_param}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class SpecShapesToTensor(object):
    """
    Transforms melspec ndarrays in sample to Tensors
    """

    def __call__(self, sample):
        melspec = sample['melspec']
        shape_param = sample['shape_param']

        # Convert to tensors and add single colour channel to spectograms
        melspec = torch.from_numpy(melspec).unsqueeze(0)
        shape_param = torch.from_numpy(shape_param)

        return {'melspec': melspec, 'shape_param': shape_param}

if __name__ == "__main__":
    data_path = '/home/peter/Documents/Uni/Project/src/model/data'
    mel_dir = 'spectograms'
    shape_dir = 'blendshapes'
    mel_path = os.path.join(data_path, mel_dir)
    shape_path = os.path.join(data_path, shape_dir)

    melspec_dataset = MelSpecBlendshapesDataset(mel_path, 
                                               shape_path, 
                                               transform=SpecShapesToTensor())

    #for i in range(len(melspec_dataset)):
    #    sample = melspec_dataset[i]

    #    print(i, sample['melspec'].shape, sample['shape_param'].shape)
    #    if i == 3:
    #        break
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(melspec_dataset, batch_size=4, num_workers=0)

    for idx, sample in enumerate(dataloader):
        print(idx, sample['melspec'].shape, sample['shape_param'].shape)
        if idx == 4:
            break
