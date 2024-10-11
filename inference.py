import torch
from train import FeedForwardNet, download_mnist_datasets

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
    # set model to evaluation mode
    model.eval()

    # evaluate without gradients
    # gradients are only needed during training
    with torch.no_grad():
        # pass input through model and get predictions in a 2D tensor (1, 10)
        predictions = model(input)
        # get index of predicted value (max of predictions)
        predicted_index = predictions[0].argmax(0)
        # get predicted value from class mapping
        predicted = class_mapping[predicted_index]
        # get expected value from class mapping
        expected = class_mapping[target]

    return predicted, expected


if __name__ == "__main__":
    # load the model
    feed_forward_net = FeedForwardNet()             # instantiate the model
    state_dict = torch.load("feedforwardnet.pth")   # load the state dict
    # feed the state dict into the model
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get sample from validation dataset for inference
    # take first sample (index 0), first dimension is input and second dimension is target
    input, target = validation_data[0][0], validation_data[0][1]

    # make inference
    predicted, expected = predict(
        feed_forward_net, input, target, class_mapping)

    print(f"Predicted: {predicted}")
    print(f"Expected: {expected}")
