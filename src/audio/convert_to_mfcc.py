from melfrqcep import MelFeqCep
from tqdm import tqdm
import os

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_dir = 'test_samples'
    spectogram_dir = 'test_spectograms'

    audio_files = os.path.join(dir_path, file_dir)
    spec_files = os.path.join(dir_path, spectogram_dir)

    if not os.path.exists(spec_files):
        os.makedirs(spec_files)

    mel_feq_cep = MelFeqCep(file_path=audio_files, save_path=spec_files)

    mel_feq_cep.convert_to_mfcc(nested=True)