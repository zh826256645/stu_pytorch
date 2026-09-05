import torch
from torchvision.transforms import ToTensor

from datasets import CaptchaData, alphabet
from training_metrics import calculate_accuracy_counts

from .models import DenseNet
from .train import calculate_loss

image, target = CaptchaData("./data/test", transform=ToTensor())[0]
model = DenseNet(weights=None).eval()
with torch.inference_mode():
    logits = model(image.unsqueeze(0))

assert logits.shape == (1, 4 * len(alphabet))
for criterion in (torch.nn.MultiLabelSoftMarginLoss(), torch.nn.CrossEntropyLoss()):
    assert torch.isfinite(calculate_loss(logits, target.unsqueeze(0), criterion))

expected_indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
predicted_indices = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5]])
expected = torch.nn.functional.one_hot(
    expected_indices,
    num_classes=len(alphabet),
).float()
predicted_logits = torch.nn.functional.one_hot(
    predicted_indices,
    num_classes=len(alphabet),
).float()
accuracy_counts = calculate_accuracy_counts(
    predicted_logits.reshape(2, -1),
    expected.reshape(2, -1),
    num_char=4,
    num_class=len(alphabet),
)
assert accuracy_counts.exact_match_accuracy == 0.5
assert accuracy_counts.character_accuracy == 0.75
