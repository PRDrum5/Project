import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from tqdm import tqdm

class MelSpec():
    def __init__(self, file_path=None, save_path=None):
        self.file_path = file_path
        self.save_path = save_path

        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.wav_files = []
        for root, _dirs, files in os.walk(self.file_path):
            for file in files:
                if file.endswith('.wav'):
                    self.wav_files.append(os.path.join(root, file))
        self.wav_files = sorted(self.wav_files)
        self.n_files = len(self.wav_files)

    def read_wav(self, wav_file):
        """
        Reads a wav file and normalised based on the maximum amplitude.
        """
        if not wav_file.endswith('.wav'):
            wav_file += '.wav'
        wav_file = os.path.join(self.file_path, wav_file)

        sample_rate, audio_data = wavfile.read(wav_file)
        if audio_data.max() != 0:
            audio_data = audio_data / audio_data.max()

        return sample_rate, audio_data
    
    def display_waveplot(self, audio, sample_rate):
        """
        Displays waveplot of audiofile
        """
        librosa.display.waveplot(y=audio, sr=sample_rate)
        plt.show()
    
    def display_melplot(self, mel_spec):
        """
        Display mel spectogram
        """
        librosa.display.specshow(librosa.power_to_db(mel_spec), x_axis='time')
        plt.show()
    
    def mfcc(self, audio_data, sample_rate, n_filters=30):
        """
        Returns the mel spectrum of an audio signal.
        The number of mel filters can be varied.
        """
        mel_spec = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
        return mel_spec

    def save_spectrum(self, spectrum, filename, save_path):
        """
        Exports spectrum file
        """
        save_path = os.path.join(save_path, filename)
        np.save(save_path, spectrum)
    
    def convert_to_mfcc(self, nested=False):
        """
        Converts all wav files into mel spectograms and exports them as numpy 
        array files.
        """
        for f in tqdm(self.wav_files):
            if nested:
                folder = f.split(os.sep)[-2]
                file_save_path = os.path.join(self.save_path, folder)

                if not os.path.exists(file_save_path):
                    os.makedirs(file_save_path)
            else:
                file_save_path = self.save_path

            filename = os.path.basename(f)
            filename = os.path.splitext(filename)[0]

            sr, audio_data = self.read_wav(f)
            mel_spec = self.mfcc(audio_data, sr)
            self.save_spectrum(mel_spec, filename, file_save_path)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = 'samples'
    spectogram_path = 'spectograms'
    file_name = 'sentence01'

    audio_files = os.path.join(dir_path, file_path)
    spec_files = os.path.join(dir_path, spectogram_path)

    ms = MelSpec(file_path=dir_path, save_path=dir_path)
    #ms.convert_to_mfcc()
    sr, audio_data = ms.read_wav(file_name)
    mel_spec = ms.mfcc(audio_data, sr)
    print(mel_spec.shape)
    ms.display_melplot(mel_spec)
    #ms.save_spectrum(mel_spec, file_name)