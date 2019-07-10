import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

def mfcc_size_pred(frames):
    """
    Best fit equation to predict the size of the mfcc given the number of video frames.
    """
    y = (frames * 1.396983933379579) - 0.9545913318493555
    y = round(y)
    return y

def gen_file_list(path, ext):
    f_list = []
    for root, _dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                f_list.append(os.path.join(root, file))
    f_list = sorted(f_list)
    return f_list

normalize = lambda array, min_val, max_val: (array - min_val) / (max_val - min_val)

standardize = lambda array, mean, std: (array - mean) / std

def mfcc_hist(f_list, title=None, verbose=False):
    sample = np.load(f_list[0])
    height, width = sample.shape

    magnitude_array = np.zeros((height*len(f_list), width))
    for idx, file in enumerate(f_list):
        temp = np.load(file)
        magnitude_array[idx*height:((idx+1)*height):,:] = temp
    magnitude_array = magnitude_array.reshape(-1,)

    plt.hist(magnitude_array, bins='auto')
    plt.title(title)
    plt.show()

    if verbose:
        min_val = magnitude_array.min()
        max_val = magnitude_array.max()
        mean = magnitude_array.mean()
        std = magnitude_array.std()
        stats = {
            'min': min_val,
            'max': max_val,
            'mean': mean,
            'std': std
        }
        return stats

def transform_data(f_list, save_dir, transform, *args):
    for file in f_list:
        save_path = file.split('/')
        save_path[-3] = save_dir
        file_name = save_path.pop()
        save_path = '/'.join(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        temp = np.load(file)
        temp = transform(temp, *args)
        np.save(os.path.join(save_path, file_name), temp)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    spec_dir = 'spectograms'
    shape_path = '/home/peter/Documents/Uni/Project/src/mesh/shape_params'
    spec_path = os.path.join(dir_path, spec_dir)
    #shape_path = os.path.join(dir_path, shape_dir)

    spec_list = gen_file_list(spec_path, ext='.npy')
    shape_list = gen_file_list(shape_path, ext='.npy')
    print(spec_list)
    print(shape_list)

    durations = np.zeros((len(spec_list),2))
    for f in range(len(spec_list)):
        spec = np.load(spec_list[f])
        spec_len = spec.shape[1]
        durations[f,0] = spec_len
        shape_p = np.load(shape_list[f])
        shape_p_len = shape_p.shape[1]
        durations[f,1] = shape_p_len
    slope, intercept = np.polyfit(durations[:,1], durations[:,0], 1)
    print(slope, intercept)
    plt.plot(durations[:,1], durations[:,0])
    plt.show()
    print(durations)


