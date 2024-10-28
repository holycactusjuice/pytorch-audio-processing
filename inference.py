import torch
import torchaudio

from cnn import CNN
from urban_sound_dataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

MODEL_PATH = "cnn.pth"

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
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


def calculate_accuracy(model, dataset, class_mapping):
    total = 0
    correct = 0
    # get sample from validation dataset for inference
    # first index is tensor object, second index is target
    for i in range(len(dataset)):
        input, target = dataset[i][0], dataset[i][1]
        input.unsqueeze_(0)
        predicted, expected = predict(model, input, target, class_mapping)
        total += 1
        if predicted == expected:
            print("Correct")
            correct += 1
        else:
            print("Incorrect")
    return round(correct / total, 6)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # load model
    cnn = CNN().to(device)
    # get the model parameters
    state_dict = torch.load(MODEL_PATH)
    # load the model with model parameters
    cnn.load_state_dict(state_dict=state_dict)

    # get validation data (not using train data)
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

    # calculate accuracy
    model_accuracy = calculate_accuracy(
        cnn, usd, class_mapping)

    print(f"Model accuracy: {model_accuracy * 100} %")
