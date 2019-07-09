import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.io import wavfile
import librosa
import numpy as np

class WavBlendshapesDataset(Dataset):
    """
    Raw Wav files (variable length)
    Blendshape Parameter Targets for each wav file.

    This is currently using all blendshape parameters
    """

    def __init__(self, wav_path, blendshapes_path, window_size=50):
        self.wav_path = wav_path
        self.blendshapes_path = blendshapes_path
        self.window_size = window_size
        self.wav_list = sorted(os.listdir(self.wav_path))
        self.stats = {'melspec_min': np.inf,
                      'melspec_max': -np.inf,
                      'shape_min': np.inf,
                      'shape_max': -np.inf}
        self.frames = 0

        self._collect_stats()

    def __len__(self):
        return len(self.wav_list)
    
    def __getitem__(self, idx):
        melspec, shape_param = self._get_data_pair(idx)

        sample = {'melspec': melspec.astype(np.float32), 
                  'shape_param': shape_param.astype(np.float32)}
        
        sample = self._normalize_to_tensor(sample)
        self._window(sample)
        
        return sample
    
    def _get_data_pair(self, idx):
        item_name = self.wav_list[idx].split('.')[0]
        wav_name = item_name + '.wav'
        shape_name = item_name + '.npy'

        sr, audio_data = wavfile.read(os.path.join(self.wav_path, wav_name))
        audio_data = audio_data / audio_data.max() 
        melspec = self._mfcc(audio_data, sr)

        shape_param = np.load(os.path.join(self.blendshapes_path, shape_name))
        return melspec, shape_param

    def _collect_stats(self):
        """
        Finds max and min values for blendshape params and melspecs for dataset.
        Finds the number of blendshape frames for the mel spectogram window.
        """
        durations = np.zeros((self.__len__(),2))

        for idx in range(self.__len__()):
            melspec, shape_param = self._get_data_pair(idx)

            melspec_len = melspec_len.shape[1]
            shape_param_len = shape_param.shape[1]

            durations[idx,0] = melspec_len
            durations[idx,1] = shape_param_len
            gradient, intercept = np.polyfit(durations[:,1], durations[:,0], 1)
            self.frames = int(round((self.window_size * gradient) + intercept))

            idx_melspec_min = melspec.min()
            idx_melspec_max = melspec.max()
            idx_shape_min = shape_param.min()
            idx_shape_max = shape_param.max()

            if idx_melspec_min < self.stats['melspec_min']: 
                self.stats['melspec_min'] = idx_melspec_min
            if idx_melspec_max > self.stats['melspec_max']: 
                self.stats['melspec_max'] = idx_melspec_max
            if idx_shape_min < self.stats['shape_min']: 
                self.stats['shape_min'] = idx_shape_min
            if idx_shape_max > self.stats['shape_max']: 
                self.stats['shape_max'] = idx_shape_max
    
    def _mfcc(self, audio_data, sample_rate, n_mfcc=50):
        """
        Returns the mel spectrum of an audio signal.
        The number of mel filters can be varied.
        """
        mel_spec = librosa.feature.mfcc(y=audio_data, 
                                        sr=sample_rate, 
                                        n_mfcc=n_mfcc)
        return mel_spec
    
    def _normalize_to_tensor(self, sample):
        """
        normalizes mel spectograms and blendshape parameters based on max and
        min values of the dataset for each.
        Then coverts these to torch tensors.
        """
        melspec = sample['melspec']
        shape_param = sample['shape_param']

        norm = lambda array, min_v, max_v: (array - min_v) / (max_v - min_v)

        melspec = norm(melspec, 
                       self.stats['melspec_min'], 
                       self.stats['melspec_max'])

        shape_param = norm(shape_param, 
                           self.stats['shape_min'], 
                           self.stats['shape_max'])
        
        melspec = torch.from_numpy(melspec).unsqueeze(0)
        shape_param = torch.from_numpy(shape_param).unsqueeze(0)

        return {'melspec': melspec, 'shape_param': shape_param}
    
    def _window(self, sample):
        melspec = sample['melspec']
        shape_param = sample['shape_param']
        #TODO slice melspec and shapes_params into windows
    



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

        sample = {'melspec': melspec.astype(np.float32), 
                  'shape_param': shape_param.astype(np.float32)}

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
        shape_param = torch.from_numpy(shape_param).unsqueeze(0)

        return {'melspec': melspec, 'shape_param': shape_param}

if __name__ == "__main__":
    data_path = '/home/peter/Documents/Uni/Project/src/model/data'
    wav_dir = 'wavs'
    shape_dir = 'blendshapes'
    wav_path = os.path.join(data_path, wav_dir)
    shape_path = os.path.join(data_path, shape_dir)

    dataset = WavBlendshapesDataset(wav_path, shape_path)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(idx, sample['melspec'].shape, sample['shape_param'].shape)
        