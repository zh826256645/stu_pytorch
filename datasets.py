import os
from PIL import Image
import torch
from torch.utils.data import Dataset

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]

alphabet = "".join(source)

alphabet = "0123456789x=+-"


def img_loader(img_path):
    img = Image.open(img_path)
    img = img.resize((180, 100))
    return img.convert("RGB")


def make_dataset(data_path, alphabet, num_class, num_char):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        target_str = img_name.split(".")[0]
        assert len(target_str) == num_char
        target = []
        for char in target_str:
            vec = [0] * num_class
            vec[alphabet.find(char)] = 1
            target += vec

        samples.append((img_path, target))
    return samples


class CaptchaData(Dataset):
    def __init__(
        self,
        data_path,
        num_class=14,
        num_char=4,
        transform=None,
        target_transform=None,
        alphabet=alphabet,
    ):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(
            self.data_path, self.alphabet, self.num_class, self.num_char
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)


# from captcha.image import ImageCaptcha
# import random
#
# list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
#         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# # 定义验证码尺寸
# width, height = 170, 80
# #生成一万张验证码
# for i in range(100):
#     generator = ImageCaptcha(width=width, height=height)
#     # 从list中取出4个字符
#     random_str = ''.join([random.choice(list) for j in range(4)])
#     # 生成验证码
#     img = generator.generate_image(random_str)
#     # 在验证码上加干扰点
#     generator.create_noise_dots(img, '#000000', 4, 40)
#     # 在验证码上加干扰线
#     generator.create_noise_curve(img, '#000000')
#     # 将图片保存在目录yzm文件夹下
#     file_name = './da/'+random_str+'_'+str(i)+'.jpg'
#     img.save(file_name)
# import nntplib as nn
# conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
