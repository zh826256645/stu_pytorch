import os

import torch
from PIL import Image
from torch.utils.data import Dataset

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
DEFAULT_NUM_CLASS = len(alphabet)
SUPPORTED_IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


def img_loader(img_path):
    img = Image.open(img_path)
    img = img.resize((180, 100))
    return img.convert("RGB")


def make_dataset(data_path, alphabet, num_class, num_char):
    samples = []
    for img_name in sorted(os.listdir(data_path)):
        img_path = os.path.join(data_path, img_name)
        suffix = os.path.splitext(img_name)[1].lower()
        if not os.path.isfile(img_path) or suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        target_str = os.path.splitext(img_name)[0]
        if len(target_str) != num_char:
            raise ValueError(f"invalid label length: {img_name}")

        target = []
        for char in target_str:
            index = alphabet.find(char)
            if index < 0:
                raise ValueError(f"invalid label character {char!r}: {img_name}")
            vec = [0] * num_class
            vec[index] = 1
            target += vec

        samples.append((img_path, target))
    return samples


class CaptchaData(Dataset):
    def __init__(
        self,
        data_path,
        num_class=DEFAULT_NUM_CLASS,
        num_char=4,
        transform=None,
        target_transform=None,
        alphabet=alphabet,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.samples = make_dataset(data_path, alphabet, num_class, num_char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.tensor(target, dtype=torch.float32)
