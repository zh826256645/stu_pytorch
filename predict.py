# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:07:17 2019

@author: icetong
"""
import os

import torch
import torch.nn as nn

# from models import CNN
from models import DenseNet
from datasets import CaptchaData
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plot

model_path = "./checkpoints/model.pth"

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]

alphabet = "0123456789x=+-"


def recognize(img, filepath, num_char=4, num_class=36):
    transforms = Compose([ToTensor()])
    img = transforms(img)
    target_str = os.path.basename(filepath).split(".")[0]
    assert len(target_str) == num_char

    target = []
    for char in target_str:
        vec = [0] * num_class
        vec[alphabet.find(char)] = 1
        target += vec

    target = torch.Tensor(target)

    cnn = DenseNet()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))

    img = img.view(1, 3, 100, 180).cuda()
    target = target.view(1, 4 * 36).cuda()
    output = cnn(img)

    output = output.view(-1, 36)
    target = target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output = output.view(-1, 4)[0]
    target = target.view(-1, 4)[0]

    return "".join([alphabet[i] for i in output.cpu().numpy()]), "".join(
        [alphabet[i] for i in target.cpu().numpy()]
    )


def predict(img_dir="./data/test2"):
    transforms = Compose([ToTensor()])
    dataset = CaptchaData(img_dir, transform=transforms)

    # cnn = CNN()
    cnn = DenseNet()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))

    total = 0
    succ = 0
    for k, (img, target) in enumerate(dataset):
        img = img.view(1, 3, 100, 180).cuda()
        target = target.view(1, 4 * 14).cuda()
        output = cnn(img)

        output = output.view(-1, 14)
        target = target.view(-1, 14)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output = output.view(-1, 4)[0]
        target = target.view(-1, 4)[0]

        pred = "".join([alphabet[i] for i in output.cpu().numpy()])
        print("pred: " + pred)
        true_str = "".join([alphabet[i] for i in target.cpu().numpy()])
        print("true: " + true_str)
        total += 1
        if pred == true_str:
            succ += 1

        plot.imshow(img.permute((0, 2, 3, 1))[0].cpu().numpy())
        plot.show()

        # if k >= 10:
        #     break

    if total:
        print(f"success rate: {round(succ / total * 100, 2)}%")


if __name__ == "__main__":
    predict()
