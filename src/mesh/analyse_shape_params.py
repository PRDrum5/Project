import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    shape_params_dir = os.path.join(dir_path, 'shape_params')
    save_dir = os.path.join(dir_path, 'aligned')

    shape_param_files = []
    for root, _dirs, files in os.walk(shape_params_dir):
        for file in files:
            if file.endswith("1.npy"):
                shape_param_files.append(os.path.join(root, file))

    n_files = len(shape_param_files)
    n_shapes = None
    frames = 0

    for file in shape_param_files:
        _params = np.load(file)
        if not n_shapes:
            n_shapes = _params.shape[0]
        else:
            assert n_shapes == _params.shape[0]
        frames += _params.shape[1]

    shape_params = np.zeros((n_shapes, frames))

    position = 0
    for file in shape_param_files:
        _params = np.load(file)
        frames_in_file = _params.shape[1]
        shape_params[:, position:position+frames_in_file] = _params
        position += frames_in_file
    
    std = np.std(shape_params, axis=1)
#    np.save(os.path.join(save_dir, 'standard_div_subject1'), std)

    std_100 = np.delete(std, range(100, n_shapes))

    y = np.arange(100)
    plt.plot(y, std_100)
    plt.title("Standard Deviation along first 100 Blendshapes")
    plt.show()

    std_10 = np.delete(std, range(10, n_shapes))

    y = np.arange(10)
    plt.plot(y, std_10)
    plt.title("Standard Deviation along first 10 Blendshapes")
    plt.show()