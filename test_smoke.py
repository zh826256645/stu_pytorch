import torch
from torchvision.transforms import ToTensor

from datasets import CaptchaData, alphabet
from models import DenseNet
from train import calculate_loss


image, target = CaptchaData("./data/test", transform=ToTensor())[0]
model = DenseNet(weights=None).eval()
with torch.inference_mode():
    logits = model(image.unsqueeze(0))

assert logits.shape == (1, 4 * len(alphabet))
for criterion in (torch.nn.MultiLabelSoftMarginLoss(), torch.nn.CrossEntropyLoss()):
    assert torch.isfinite(calculate_loss(logits, target.unsqueeze(0), criterion))
