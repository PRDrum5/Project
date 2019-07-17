import torch
from torch.utils.data import Dataset
import os
import math
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

    def __init__(self, wav_path, blendshapes_path, n_shapes=10, transform=None):
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
        mfcc, shape_param = self._get_data_pair(idx)

        sample = {'mfcc': mfcc.astype(np.float32), 
                  'shape_param': shape_param.astype(np.float32)}
        
        sample = self._normalize(sample)

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _get_data_pair(self, idx):
        item_name = self.wav_list[idx].split('.')[0]
        wav_name = item_name + '.wav'
        shape_name = item_name + '.npy'

        sr, audio_data = wavfile.read(os.path.join(self.wav_path, wav_name))
        audio_data = audio_data / audio_data.max() 
        mfcc = self._mfcc(audio_data, sr)

        shape_param = np.load(os.path.join(self.blendshapes_path, shape_name))
        crop_range = range(self.n_shapes, shape_param.shape[0])
        shape_param = np.delete(shape_param, crop_range, axis=0)

        return mfcc, shape_param

    def _collect_stats(self):
        """
        Finds max and min values for blendshape params and mfcc for dataset.
        """
        for idx in range(self.__len__()):
            mfcc, shape_param = self._get_data_pair(idx)

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
    
    def _mfcc(self, audio_data, sample_rate, n_mfcc=50):
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
        min_v = self.stats['shape_min']
        max_v = self.stats['shape_max']
        denormed = (shape_param * (max_v - min_v)) + min_v

        return denormed

class DropFramesToMfccDuration(object):
    """
    Randomly drops frames from blendshape parameters so that MFCC and shape 
    params have same length
    """

    def __call__(self, sample):
        mfcc = sample['mfcc']
        shape_param = sample['shape_param']

        mfcc_len = mfcc.size(1)
        shape_param_len = shape_param.size(1)
        len_diff = shape_param_len - mfcc_len

        random_drop = torch.rand(1, len_diff)
        1/0
        return sample

class MergeFrameToMfccDuration(object):
    """
    Randomly merges adjacent frames from blendshape params into the mean of the 
    two so that MFCC and shape params have the same length
    """

    def __call__(self, sample):
        mfcc = sample['mfcc']
        shape_param = sample['shape_param']

        shape_param = self.frame_merger(mfcc, shape_param)

        sample = {'mfcc': mfcc, 'shape_param': shape_param}

        return sample
    
    def frame_merger(self, target, current):
        target_len = target.shape[1]
        current_len = current.shape[1]
        assert target_len < current_len, "Target length must be less \
                                          than current length"
        diff = current_len - target_len

        step = int(math.ceil(current_len / diff)) # Round up step
        reduction = ((current_len - (current_len % step)) / step)
        reduced_len = current_len - reduction

        # Prvents first frame always being reduced
        random_start = np.random.randint(0, step-1)
        idx_to_merge = np.arange(random_start, current_len-1, step)
        for idx in idx_to_merge:
            mean = (current[:, idx] + current[:, idx+1]) / 2
            current[:, idx] = mean
        
        current = np.delete(current, idx_to_merge+1, axis=1)

        current_len = current.shape[1]
        diff = current_len - target_len

        if diff > 0:
            current = self.frame_merger(target, current)
        
        return current
    
class RandomOneSecondMfccCrop(object):
    """
    Takes a random one second crop from the MFCC and corresponding shape params
    """
    
    def __call__(self, sample):
        mfcc = sample['mfcc']
        shape_param = sample['shape_param']

        mfcc_len = mfcc.shape[1]
        shape_param_len = shape_param.shape[1]
        assert mfcc_len == shape_param_len, "MFCC and Shapes must \
                                             have same length"

        clip_duration = mfcc_len
        one_sec_duration = 43
        latest_start = clip_duration - one_sec_duration

        random_start = np.random.randint(0, latest_start)
        end_point = random_start + one_sec_duration

        mfcc = mfcc[:, random_start:end_point]
        shape_param = shape_param[:, random_start:end_point]

        sample = {'mfcc': mfcc, 'shape_param': shape_param}
        return sample

class OneSecondMfccCrop(object):
    """
    Takes the first second of each clip
    """
    
    def __call__(self, sample):
        mfcc = sample['mfcc']
        shape_param = sample['shape_param']

        mfcc_len = mfcc.shape[1]
        shape_param_len = shape_param.shape[1]
        assert mfcc_len == shape_param_len, "MFCC and Shapes must \
                                             have same length"

        clip_duration = mfcc_len
        one_sec_duration = 43
        latest_start = clip_duration - one_sec_duration

        mfcc = mfcc[:, 0:one_sec_duration]
        shape_param = shape_param[:, 0:one_sec_duration]

        sample = {'mfcc': mfcc, 'shape_param': shape_param}
        return sample

class MFCCBlendshapesDataset(Dataset):
    """
    mfcc features
    Blendshape Parameters targets
    """

    def __init__(self, mfcc_dir, blendshapes_dir, transform=None):
        self.mfcc_dir = mfcc_dir
        self.blendshapes_dir = blendshapes_dir
        self.transform = transform
        self.mfcc_list = sorted(os.listdir(mfcc_dir))

    def __len__(self):
        _path, _dirs, files = next(os.walk(self.mfcc_dir))
        length = len(files)
        return length
    
    def __getitem__(self, idx):
        item_name = self.mfcc_list[idx]

        mfcc = np.load(os.path.join(self.mfcc_dir, item_name))
        shape_param = np.load(os.path.join(self.blendshapes_dir, item_name))

        sample = {'mfcc': mfcc.astype(np.float32), 
                  'shape_param': shape_param.astype(np.float32)}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

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

if __name__ == "__main__":
    data_path = '/home/peter/Documents/Uni/Project/src/model/data'
    wav_dir = 'wavs'
    shape_dir = 'blendshapes'
    wav_path = os.path.join(data_path, wav_dir)
    shape_path = os.path.join(data_path, shape_dir)

    dataset = WavBlendshapesDataset(wav_path, shape_path)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(idx, sample['mfcc'].shape, sample['shape_param'].shape)
        