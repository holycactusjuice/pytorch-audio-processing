# tutorial: https://www.youtube.com/watch?v=4p0G6tgNLis&t=549s&ab_channel=ValerioVelardo-TheSoundofAI

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

######## STEPS ########
# 1. download dataset
# 2. create data loader
# 3. build model
# 4. train
# 5. save trained model
#######################

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "feedforwardnet.pth"


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        ### DEFINING LAYERS OF THE NEURAL NETWORK ###
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(  # sequential allows us to pack multiple layers together sequentially
            nn.Linear(28*28, 256),
            # first param is number of input features, second param is number of output features
            # 28*28 for the number of pixels in a 28 x 28 grid
            # 256 output features is just an arbitrary number
            # we can tweak the number of output features and hidden layers
            nn.ReLU(),
            nn.Linear(256, 10),
            # 10 since there are 10 digits
        )
        # takes the output feature activations and normalizes them so that they sum to 1
        self.softmax = nn.Softmax(dim=1)

    # this function is implicitly called under the hood by PyTorch during training
    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        # data will be stored in a new directory "data" in the working directory
        root="data",
        download=True,          # we want to download the data
        train=True,             # we want the training part of the data
        # converts all the inputs to tensors; all values normalized between 0 and 1
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,             # we do not want the training part of the data
        transform=ToTensor()
    )
    return train_data, validation_data


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
    # download MNIST dataset
    train_data, validation_data = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create data loader for the train dataset
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # check if cuda is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # build model using available device
    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_data_loader,
          loss_fn, optimizer, device, EPOCHS)

    # save the model
    # state_dict() is a Python dict with all the info about layers and parameters
    torch.save(feed_forward_net.state_dict(), MODEL_PATH)
    print(f"Model trained and stored at {MODEL_PATH}")
