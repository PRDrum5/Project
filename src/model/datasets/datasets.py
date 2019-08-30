import torch
import torch.tensor as tensor
from torch.utils.data import Dataset
import os
import math
import numpy as np
from scipy.io import wavfile
import librosa
import numpy as np
from tqdm import tqdm
import pickle


class LrwBlendshapesDataset(Dataset):
    """
    Blendshape Parameters for audio clips from the BBC LRW words dataset,
    processed by the VOCA model.

    500 words (labels), 43 frames for 1 second, 4 shape parameters per frame.
    """
    def __init__(self, blendshapes_path, transform=None):
        self.blendshapes_path = blendshapes_path
        self.blendshapes_list = sorted(os.listdir(self.blendshapes_path))
        self.transform = transform

        self.stats = {'shape_min': np.inf,
                      'shape_max': -np.inf}

        self._collect_stats()

    def __len__(self):
        return len(self.blendshapes_list)
    
    def __getitem__(self, idx):
        """
        idx references the sample number
        """
        shape_params, label = self._get_sample(idx)

        sample = {'shape_params': shape_params.astype(np.float32),
                  'label': label.astype(np.int64)}

        sample = self._normalize(sample)

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _get_sample(self, idx):
        """
        fetches the relevant sample
        """
        sample_file = self.blendshapes_list[idx]
        sample_name = sample_file.split('.')[0]
        word_label, word_id = sample_name.split('_')
        word_label = np.array(int(word_label))
        shape_params = np.load(os.path.join(self.blendshapes_path, sample_file))
        return shape_params, word_label


    def _collect_stats(self):
        """
        Finds max and min values for blendshape params for dataset.
        """
        try:
            with open('data/lrw_shape_stats.pkl', 'rb') as f:
                self.stats = pickle.load(f)
        except:
            print("Collecting dataset statistics LRW...\n")
            for idx in tqdm(range(self.__len__())):
                shape_param, _label = self._get_sample(idx)

                idx_shape_min = shape_param.min()
                idx_shape_max = shape_param.max()

                if idx_shape_min < self.stats['shape_min']: 
                    self.stats['shape_min'] = idx_shape_min
                if idx_shape_max > self.stats['shape_max']: 
                    self.stats['shape_max'] = idx_shape_max

            with open('data/lrw_shape_stats.pkl', 'wb') as f:
                pickle.dump(self.stats, f, pickle.HIGHEST_PROTOCOL)

    def _normalize(self, sample):
        """
        normalizes blendshape parameters based on max and
        min values of the dataset.
        """
        label = sample['label']
        shape_params = sample['shape_params']

        norm = lambda array, min_v, max_v: (array - min_v) / (max_v - min_v)

        shape_params = norm(shape_params, 
                           self.stats['shape_min'], 
                           self.stats['shape_max'])

        return {'shape_params': shape_params,
                'label': label}


class WavBlendshapesDataset(Dataset):
    """
    Raw Wav files (variable length)
    Blendshape Parameter Targets for each wav file.

    This is currently using all blendshape parameters
    """

    def __init__(self, wav_path, blendshapes_path, n_shapes=4, transform=None):
        self.wav_path = wav_path
        self.blendshapes_path = blendshapes_path
        self.n_shapes = n_shapes
        self.wav_list = sorted(os.listdir(self.wav_path))
        self.transform = transform

        self.stats = {'mfcc_min': np.inf,
                      'mfcc_max': -np.inf,
                      'shape_min': np.inf,
                      'shape_max': -np.inf}

        self._collect_stats()

    def __len__(self):
        return len(self.wav_list)
    
    def __getitem__(self, idx):
        mfcc, shape_param, item_name = self._get_data_pair(idx)

        sample = {'mfcc': mfcc.astype(np.float32), 
                  'shape_param': shape_param.astype(np.float32)}
        
        sample = self._normalize(sample)

        if self.transform:
            sample = self.transform(sample)
        
        sample['item_name'] = item_name
        
        return sample
    
    def _get_data_pair(self, idx):
        item_name = self.wav_list[idx].split('.')[0]
        wav_name = item_name + '.wav'
        shape_name = item_name + '.npy'

        sr, audio_data = wavfile.read(os.path.join(self.wav_path, wav_name))

        if len(audio_data.shape) > 1:
            audio_data = np.delete(audio_data, 1, 1).reshape(-1,)

        audio_data = audio_data / audio_data.max() 
        mfcc = self._mfcc(audio_data, sr)

        shape_param = np.load(os.path.join(self.blendshapes_path, shape_name))
        crop_range = range(self.n_shapes, shape_param.shape[0])
        shape_param = np.delete(shape_param, crop_range, axis=0)

        return mfcc, shape_param, item_name

    def _collect_stats(self):
        """
        Finds max and min values for blendshape params and mfcc for dataset.
        """
        try:
            with open('data/lrw_audio_stats.pkl', 'rb') as f:
                self.stats = pickle.load(f)
        except:
            print("Collecting dataset statistics...\n")
            for idx in tqdm(range(self.__len__())):
                mfcc, shape_param, _item_name = self._get_data_pair(idx)

                idx_mfcc_min = mfcc.min()
                idx_mfcc_max = mfcc.max()

                idx_shape_min = shape_param.min()
                idx_shape_max = shape_param.max()

                if idx_mfcc_min < self.stats['mfcc_min']: 
                    self.stats['mfcc_min'] = idx_mfcc_min
                if idx_mfcc_max > self.stats['mfcc_max']: 
                    self.stats['mfcc_max'] = idx_mfcc_max

                if idx_shape_min < self.stats['shape_min']:
                    self.stats['shape_min'] = idx_shape_min
                if idx_shape_max > self.stats['shape_max']:
                    self.stats['shape_max'] = idx_shape_max

            with open('data/lrw_audio_stats.pkl', 'wb') as f:
                pickle.dump(self.stats, f, pickle.HIGHEST_PROTOCOL)
            
    def _mfcc(self, audio_data, sample_rate, n_mfcc=12):
        """
        Returns the mfcc of an audio signal.
        The number of mfcc filters can be varied.
        """

        mfcc = librosa.feature.mfcc(y=audio_data, 
                                    sr=sample_rate, 
                                    n_mfcc=n_mfcc)

        return mfcc
    
    def _fix_array_width(self, array, target_width):
        """
        Crops or pads (zeros) array to be a given width
        """
        height, width = array.shape

        if width > target_width:
            crop_range = range(target_width, width)
            array = np.delete(array, crop_range, axis=1) 
        else:
            padded_array = np.zeros((height, target_width))
            padded_array[:height,:width] = array
            array = padded_array
        return array

    
    def _normalize(self, sample):
        """
        normalizes mfcc and blendshape parameters based on max and
        min values of the dataset for each.
        """
        mfcc = sample['mfcc']
        shape_param = sample['shape_param']

        norm = lambda array, min_v, max_v: (array - min_v) / (max_v - min_v)

        mfcc = norm(mfcc, 
                    self.stats['mfcc_min'], 
                    self.stats['mfcc_max'])
        
        shape_param = norm(shape_param,
                           self.stats['shape_min'],
                           self.stats['shape_max'])

        return {'mfcc': mfcc, 'shape_param': shape_param}
    
    def denorm(self, shape_param):
        """
        denormalizes generated blendshape parameters
        """

        _denorm = lambda array, min_v, max_v: (array * (max_v - min_v)) + min_v

        min_vals = self.stats['shape_min']
        max_vals = self.stats['shape_max']
        shape_param = _denorm(shape_param,
                              min_vals,
                              max_vals)

        return shape_param


class SpecShapesToTensor(object):
    """
    Transforms mfcc ndarrays in sample to Tensors
    """

    def __call__(self, sample):
        mfcc = sample['mfcc']
        shape_param = sample['shape_param']

        # Convert to tensors and add single colour channel to spectograms
        mfcc = torch.from_numpy(mfcc).unsqueeze(0)
        # Convert each blendshape params into channel
        shape_param = torch.from_numpy(shape_param).unsqueeze(1)

        return {'mfcc': mfcc, 'shape_param': shape_param}


class LrwShapesToTensor(object):
    """
    Transforms blendshape params to Tensors
    """
    def __call__(self, sample):
        label = sample['label']
        shape_params = sample['shape_params']

        label = torch.from_numpy(label)

        shape_params = torch.from_numpy(shape_params).unsqueeze(0)

        return {'label': label, 'shape_params': shape_params}
