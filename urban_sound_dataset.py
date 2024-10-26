# had to install the soundfile backend
# https://stackoverflow.com/questions/78097861/how-to-solve-runtimeerror-couldnt-find-appropriate-backend-to-handle-uri-in-py

import os

from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        # want to return the number of samples in the dataset
        return len(self.annotations)

    # allows us to index the dataset
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # signal : (num_channels, samples)
        signal, sr = torchaudio.load(audio_sample_path)
        # different signals have different sample rates,
        # but we want the mel spectrograms to be uniform in shape
        signal = self._resample_if_necessary(signal, sr)
        # signal may be stereo, so we convert to mono
        signal = self._mix_down_if_necessary(signal)
        # returns the mel spectrogram of signal
        signal = self.transformation(signal)
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        # only resample if the sr is different from the target sr
        if (sr != self.target_sample_rate):
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # only mix down if there's more than 1 channel
        if (signal.shape[0] > 1):  # signal.shape : e.g. (2, 1000)
            # setting dim=0 to take the mean across the 0th column
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # construct the file path
        # get the folder number
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold,
                            self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:\\Datasets\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
    AUDIO_DIR = "C:\\Datasets\\UrbanSound8K\\audio"
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR,
                            mel_spectrogram, SAMPLE_RATE)

    print(f"There are {len(usd)} samples in the dataset")

    signal, label = usd[0]
