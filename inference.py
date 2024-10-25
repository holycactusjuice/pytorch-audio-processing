# tutorial: https://www.youtube.com/watch?v=0Q5KTt2R5w4&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm&index=3&ab_channel=ValerioVelardo-TheSoundofAI

import torch
from train import FeedForwardNet, download_mnist_datasets

MODEL_PATH = "feedforwardnet.pth"

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


def predict(model, input, target, class_mapping):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():  # we don't calculate gradients when making inferences
        # returns tensor where first dimension is amount of input data and second dimension is number of output features
        predictions = model(input)
        # argmax returns the index of the greatest value
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


def calculate_accuracy(model, validation_data, class_mapping):
    total = 0
    correct = 0
    # get sample from validation dataset for inference
    # first index is tensor object, second index is target
    for input, target in validation_data:
        predicted, expected = predict(model, input, target, class_mapping)
        total += 1
        if predicted == expected:
            print("Correct")
            correct += 1
        else:
            print("Incorrect")
    return round(correct / total, 6)


if __name__ == "__main__":
    # load model
    feed_forward_net = FeedForwardNet()
    # get the model parameters
    state_dict = torch.load(MODEL_PATH)
    # load the model with model parameters
    feed_forward_net.load_state_dict(state_dict=state_dict)

    # load MNIST validation dataset
    # get validation data (not using train data)
    _, validation_data = download_mnist_datasets()

    # calculate accuracy
    model_accuracy = calculate_accuracy(
        feed_forward_net, validation_data, class_mapping)

    print(f"Model accuracy: {model_accuracy * 100} %")
