import os

import matplotlib.pyplot as plot
import torch
from torchvision.transforms import Compose, ToTensor

from datasets import CaptchaData, alphabet
from .models import DenseNet

model_path = "./stage1_densenet121/checkpoints/model-36.pth"
num_char = 4
num_class = len(alphabet)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_model():
    model = DenseNet(num_class=num_class, num_char=num_char, weights=None)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def recognize(img, filepath):
    img = Compose([ToTensor()])(img)
    target_str = os.path.basename(filepath).split(".")[0]
    if len(target_str) != num_char or any(char not in alphabet for char in target_str):
        raise ValueError(f"invalid image label: {target_str!r}")

    model = load_model()
    with torch.inference_mode():
        output = model(img.view(1, 3, 100, 180).to(device))
    output = output.view(-1, num_class).argmax(dim=1).view(-1, num_char)[0]
    prediction = "".join(alphabet[index] for index in output.cpu().numpy())
    return prediction, target_str


def predict(img_dir="./data/test"):
    dataset = CaptchaData(img_dir, transform=Compose([ToTensor()]))
    model = load_model()
    total = 0
    correct = 0

    for img, target in dataset:
        img = img.view(1, 3, 100, 180).to(device)
        with torch.inference_mode():
            output = model(img)

        output = output.view(-1, num_class).argmax(dim=1).view(-1, num_char)[0]
        target = target.view(-1, num_class).argmax(dim=1).view(-1, num_char)[0]
        prediction = "".join(alphabet[index] for index in output.cpu().numpy())
        target_str = "".join(alphabet[index] for index in target.cpu().numpy())
        print(f"pred: {prediction}")
        print(f"true: {target_str}")
        total += 1
        correct += prediction == target_str

        plot.imshow(img.permute((0, 2, 3, 1))[0].cpu().numpy())
        plot.show()

    if total:
        print(f"success rate: {round(correct / total * 100, 2)}%")


if __name__ == "__main__":
    predict()
