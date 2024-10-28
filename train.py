import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

import time

from urban_sound_dataset import UrbanSoundDataset
from cnn import CNN

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "cnn.pth"

ANNOTATIONS_FILE = "C:\\Datasets\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
AUDIO_DIR = "C:\\Datasets\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:  # loop through all the samples in the dataset
        inputs, targets = inputs.to(device), targets.to(
            device)  # assign inputs and targets to device

        ### CALCULATE LOSS ###
        # first get predictions from the model
        predictions = model(inputs)
        # calculate loss with the loss function by comparing to the targets
        loss = loss_fn(predictions, targets)

        ### BACKPROPAGATION WITH GRADIENT DESCENT ###
        optimizer.zero_grad()   # clear all previously calculated graidents
        loss.backward()         # apply backpropagation
        optimizer.step()        # update weights

    # print loss for last batch
    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    total_start_time = time.time()
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        epoch_start_time = time.time()
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print(
            f"Time taken: {round(time.time() - epoch_start_time, 2)} seconds")
        print("-----------------------")
    print("Training complete")
    print(
        f"Total time taken: {round(time.time() - total_start_time, 2)} seconds")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # instantiate dataset object and create object loader
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
        NUM_SAMPLES,
        device
    )

    # create data loader for the train dataset
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    # check if cuda is available

    # build model using available device
    cnn = CNN().to(device)
    print(cnn)

    # instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader,
          loss_fn, optimizer, device, EPOCHS)

    # save the model
    # state_dict() is a Python dict with all the info about layers and parameters
    torch.save(cnn.state_dict(), MODEL_PATH)
    print(f"Model trained and stored at {MODEL_PATH}")
