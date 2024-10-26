# had to install the soundfile backend
# https://stackoverflow.com/questions/78097861/how-to-solve-runtimeerror-couldnt-find-appropriate-backend-to-handle-uri-in-py

import os

from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

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
        # ensure that signal has samples = num_samples
        # 1. too short -> right pad with 0
        # 2. too long -> truncate
        # 3. correct len -> do nothing
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # returns the mel spectrogram of signal
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal: Tensor (1, num_samples)
        if (signal.shape[1] > self.num_samples):
            # we can list slice with a Tensor and with a 2D list!!
            # keep first dimension the same
            # slice second dimension at num_samples
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if (signal.shape[1] < self.num_samples):
            num_missing_samples = self.num_samples - signal.shape[1]
            # padding is in the format: (left_padding_last_dim, right_padding_last_dim, left_padding_2nd_last_dim, right_padding_2nd_last_dim, etc.)
            # (left_padding, right_padding)
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

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
    # sample rate is how many samples per second
    SAMPLE_RATE = 22050
    # num samples is how many samples to collect in total
    NUM_SAMPLES = 22050
    # so duration of all samples = NUM_SAMPLES / SAMPLE_RATE

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES
    )

    print(f"There are {len(usd)} samples in the dataset")

    signal, label = usd[1]
