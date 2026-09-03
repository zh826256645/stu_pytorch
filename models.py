from torch import nn
from torchvision import models


class DenseNet(nn.Module):
    def __init__(
        self, num_class=36, num_char=4, weights=models.DenseNet121_Weights.DEFAULT
    ):
        super().__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.densenet = models.densenet121(weights=weights)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_class * num_char)

    def forward(self, x):
        return self.densenet(x)
