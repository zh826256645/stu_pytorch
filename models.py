import torch.nn as nn


# class CNN(nn.Module):
#     def __init__(self, num_class=36, num_char=4):
#         super(CNN, self).__init__()
#         self.num_class = num_class
#         self.num_char = num_char
#         self.conv = nn.Sequential(
#                 #batch*3*180*100
#                 nn.Conv2d(3, 16, 3, padding=(1, 1)),
#                 nn.MaxPool2d(2, 2),
#                 nn.BatchNorm2d(16),
#                 nn.ReLU(),
#                 #batch*16*90*50
#                 nn.Conv2d(16, 64, 3, padding=(1, 1)),
#                 nn.MaxPool2d(2, 2),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(),
#                 #batch*64*45*25
#                 nn.Conv2d(64, 512, 3, padding=(1, 1)),
#                 nn.MaxPool2d(2, 2),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(),
#                 #batch*512*22*12
#                 nn.Conv2d(512, 512, 3, padding=(1, 1)),
#                 nn.MaxPool2d(2, 2),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(),
#                 #batch*512*11*6
#                 )
#         self.fc = nn.Linear(512*11*6, self.num_class*self.num_char)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(-1, 512*11*6)
#         x = self.fc(x)
#         return x

# # CRNN
# import torch.nn as nn
# import torch.nn.functional as F
#
# class CNN(nn.Module):
#     def __init__(self, num_class=36, num_char=4, rnn_hidden_size=256):
#         super(CNN, self).__init__()
#         self.num_class = num_class
#         self.num_char = num_char
#         self.rnn_hidden_size = rnn_hidden_size
#
#         # CNN部分用于特征提取
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             nn.Conv2d(16, 64, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             nn.Conv2d(64, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#
#             nn.Conv2d(512, rnn_hidden_size, 2),  # 最后一层卷积输出通道数应与RNN隐藏层大小相同
#             nn.BatchNorm2d(rnn_hidden_size),
#             nn.ReLU()
#         )
#
#         # RNN部分用于序列建模
#         self.rnn = nn.Sequential(
#             nn.LSTM(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, num_layers=2, bidirectional=True)
#         )
#
#         # 转录层(转录网络)
#         self.fc = nn.Linear(rnn_hidden_size * 2, num_class)  # 乘以2是因为LSTM是双向的
#
#     def forward(self, x):
#         # CNN部分
#         x = self.conv(x)
#
#         # 维度变换: (batch, channels, height, width) -> (width, batch, height * channels)
#         # LSTM期望输入的维度是(seqlen, batch, input_size)
#         x = x.permute(3, 0, 2, 1).contiguous()
#         x = x.view(x.size(0), x.size(1), -1)
#
#         # RNN部分
#         x, _ = self.rnn(x)
#
#         # 转录层
#         x = self.fc(x)
#
#         # 使用log_softmax获取输出概率分布
#         x = F.log_softmax(x, dim=2)
#
#         return x

# import torch.nn as nn
# from torchvision.models import vgg16
#
# class VGG(nn.Module):
#     def __init__(self, num_class=36, num_char=4):
#         super(VGG, self).__init__()
#         self.num_class = num_class
#         self.num_char = num_char
#         self.vgg = vgg16(pretrained=True)
#         self.vgg.classifier[-1] = nn.Linear(4096, self.num_class*self.num_char)
#
#     def forward(self, x):
#         features = self.vgg.features(x)
#         features = features.view(-1, 512*11*6)  # 注意调整这里的维度
#         output = self.vgg.classifier(features)
#         return output
#

# res18
# import torch
# import torch.nn as nn
# import torchvision.models as models
#
# class CNN(nn.Module):
#     def __init__(self, num_class=36, num_char=4):
#         super(CNN, self).__init__()
#         self.num_class = num_class
#         self.num_char = num_char
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.resnet.fc = nn.Linear(512, self.num_class*self.num_char)
#
#     def forward(self, x):
#         x = self.resnet(x)
#         return x
#

import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_class=14, num_char=4):
        super(DenseNet, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, self.num_class * self.num_char)

    def forward(self, x):
        x = self.densenet(x)
        return x


