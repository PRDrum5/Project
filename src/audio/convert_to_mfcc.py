from melspec import MelSpec
from tqdm import tqdm
import os

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = 'trimmed_samples'
    spectogram_path = 'spectograms'

    audio_files = os.path.join(dir_path, file_path)
    spec_files = os.path.join(dir_path, spectogram_path)

    ms = MelSpec(file_path=audio_files, save_path=spec_files)

    ms.convert_to_mfcc(nested=True)