import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from tqdm import tqdm

class MelFeqCep():
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
    
    def display_melplot(self, mfcc):
        """
        Display mfcc
        """
        librosa.display.specshow(librosa.power_to_db(mfcc), x_axis='time')
        plt.show()
    
    def mfcc(self, audio_data, sample_rate, n_mfcc=50):
        """
        Returns the mfcc of an audio signal.
        The number of mel filters can be varied.
        """
        mfcc = librosa.feature.mfcc(y=audio_data, 
                                    sr=sample_rate, 
                                    n_mfcc=n_mfcc)
        return mfcc

    def save_mfcc(self, mfcc, filename, save_path):
        """
        Exports spectrum file
        """
        save_path = os.path.join(save_path, filename)
        np.save(save_path, mfcc)
    
    def convert_to_mfcc(self, nested=False):
        """
        Converts all wav files into mfcc and exports them as numpy 
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
            mfcc = self.mfcc(audio_data, sr)
            self.save_mfcc(mfcc, filename, file_save_path)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = 'samples'
    spectogram_path = 'spectograms'
    file_name = 'sentence01'

    audio_files = os.path.join(dir_path, file_path)
    spec_files = os.path.join(dir_path, spectogram_path)

    mel_feq_cep = MelFeqCep(file_path=audio_files, save_path=dir_path)
    #mel_feq_cep.convert_to_mfcc()
    sr, audio_data = mel_feq_cep.read_wav(file_name)
    five_sec_length = 5 * sr
    print(five_sec_length)
    padded_audio = np.zeros((five_sec_length,))
    padded_audio[:audio_data.shape[0],] = audio_data
    mfcc = mel_feq_cep.mfcc(padded_audio, sr)
    print(mfcc.shape)
    mel_feq_cep.display_melplot(mfcc)
    #mel_feq_cep.save_spectrum(mfcc, file_name)