import numpy as np
import matplotlib.pyplot as plt
import os

def gen_npy_file_list(path):
    f_list = []
    for root, _dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                f_list.append(os.path.join(root, file))
    return f_list

normalize = lambda array, min_val, max_val: (array - min_val) / (max_val - min_val)

standardize = lambda array, mean, std: (array - mean) / std

def mel_spec_hist(f_list, title=None, verbose=False):
    magnitude_array = np.zeros((50*len(f_list), 5))
    for idx, file in enumerate(f_list):
        temp = np.load(file)
        magnitude_array[idx*50:((idx+1)*50):,:] = temp
    magnitude_array = magnitude_array.reshape(-1,)

    plt.hist(magnitude_array, bins='auto')
    plt.title(title)
    plt.show()

    if verbose:
        min_val = magnitude_array.min()
        max_val = magnitude_array.max()
        mean = magnitude_array.mean()
        std = magnitude_array.std()
        return min_val, max_val, mean, std

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
    normed_dir = 'normalized_spectograms'
    standard_dir = 'standardized_spectograms'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    spec_dirs = os.path.join(dir_path, 'spectograms')
    norm_spec_dirs = os.path.join(dir_path, normed_dir)
    standard_spec_dirs = os.path.join(dir_path, standard_dir)

    specs = gen_npy_file_list(spec_dirs)
    title = "Histogram of Raw Mel Spectrum Magnitudes"
    min_val, max_val, mean, std = mel_spec_hist(specs, title, verbose=True)

    transform_data(specs, normed_dir, normalize, min_val, max_val)
    norm_specs = gen_npy_file_list(norm_spec_dirs)
    title = "Histogram of Normalized Mel Spectrum Magnitudes"
    mel_spec_hist(norm_specs, title)

    transform_data(specs, standard_dir, standardize, mean, std)
    standard_specs = gen_npy_file_list(standard_spec_dirs)
    title = "Histogram of Standardized Mel Spectrum Magnitudes"
    mel_spec_hist(standard_specs, title)
