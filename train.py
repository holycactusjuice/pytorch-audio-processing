# tutorial: https://www.youtube.com/watch?v=4p0G6tgNLis&t=549s&ab_channel=ValerioVelardo-TheSoundofAI

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

######## STEPS ########
# 1. download dataset
# 2. create data loader
# 3. build model
# 4. train
# 5. save trained model
#######################

BATCH_SIZE = 128


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        ### DEFINING LAYERS OT THE NEURAL NETWORK ###
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(  # sequential allows us to pack multiple layers together sequentially
            nn.Linear(28*28, 256),
            # first param is number of input features, second param is number of output features
            # 28*28 for the number of pixels in a 28 x 28 grid
            nn.ReLU(),
            nn.Linear(256, 10),
            # 10 since there are 10 digits
        )
        # takes the output feature activations and normalizes them so that they sum to 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)


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
