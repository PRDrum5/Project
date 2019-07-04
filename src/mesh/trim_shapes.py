import numpy as np
import os
from tqdm import tqdm

def split_shape_parms(file_path, save_path, param_length, n_shapes=10):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    param_length = int(param_length)

    params = np.load(file_path)

    # Drop all but first 10 blendshape parms
    total_shapes, total_frames = params.shape
    params = np.delete(params, range(n_shapes, total_shapes), axis=0)

    n_splits = total_frames // param_length

    for file_num in range(n_splits):
        p_split = params[:,:param_length]

        filename = 'shapes' + '%02d' % (file_num+1)
        file_save_path = os.path.join(save_path, filename)
        np.save(file_save_path, p_split)

        params = np.delete(params, range(0, param_length), axis=1)

if __name__ == "__main__":

    fps = 60 # camera recorded at 60fps
    mel_spec_time = 0.1 # Mel spec are of 0.1 second durations 
    param_length = fps * mel_spec_time
    n_shapes = 10

    dir_path = os.path.dirname(os.path.realpath(__file__))

    for sentence_num in tqdm(range(1, 41)):
        shapes_dir = os.path.join(dir_path, 'shape_params')
        sentence = 'sentence' + '%02d' % sentence_num
        save_path = os.path.join(shapes_dir, sentence)
        file_name = 'shape_params_sentence' + '%02d' % sentence_num + '.npy'
        file_path = os.path.join(shapes_dir, file_name)

        split_shape_parms(file_path, save_path, param_length)
