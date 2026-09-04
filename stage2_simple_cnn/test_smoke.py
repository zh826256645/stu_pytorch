"""简单 CNN 的最小冒烟测试。

运行方式：
    uv run python -m stage2_simple_cnn.test_smoke
"""

import torch
from torchvision.transforms import Compose, RandomAffine, ToTensor

from datasets import CaptchaData, alphabet
from training_metrics import calculate_accuracy_counts

from .models import SimpleCaptchaCNN
from .train import build_image_transform, build_lr_scheduler, calculate_loss

# 读取一张真实验证码，检查项目数据能否正常送入模型。
image, target = CaptchaData("./data/test", transform=ToTensor())[0]
assert image.shape == (3, 100, 180)
assert target.shape == (4 * len(alphabet),)

# 训练增强只增加轻量仿射变换，验证预处理仍保持确定性。
train_transform = build_image_transform(augment=True)
validation_transform = build_image_transform(augment=False)
assert isinstance(train_transform, Compose)
assert isinstance(train_transform.transforms[0], RandomAffine)
assert isinstance(train_transform.transforms[1], ToTensor)
assert len(validation_transform.transforms) == 1
assert isinstance(validation_transform.transforms[0], ToTensor)
augmented_image, _ = CaptchaData("./data/test", transform=train_transform)[0]
assert augmented_image.shape == image.shape

# 验证 loss 连续不改善时，调度器会把学习率降低到原来的 30%。
scheduler_model = torch.nn.Linear(1, 1)
scheduler_optimizer = torch.optim.Adam(scheduler_model.parameters(), lr=0.001)
scheduler = build_lr_scheduler(scheduler_optimizer)
for validation_loss in (1.0, 1.1, 1.1, 1.1, 1.1):
    scheduler.step(validation_loss)
assert abs(scheduler_optimizer.param_groups[0]["lr"] - 0.0003) < 1e-12

# 增加一个批量维度：[3, 100, 180] -> [1, 3, 100, 180]。
images = image.unsqueeze(0)
targets = target.unsqueeze(0)

model = SimpleCaptchaCNN()
convolution_layers = [
    layer for layer in model.features if isinstance(layer, torch.nn.Conv2d)
]
assert [layer.out_channels for layer in convolution_layers] == [16, 32, 64]
batch_norm_layers = [
    layer for layer in model.features if isinstance(layer, torch.nn.BatchNorm2d)
]
assert [layer.num_features for layer in batch_norm_layers] == [16, 32, 64]
assert isinstance(model.classifier, torch.nn.Sequential)
assert isinstance(model.classifier[0], torch.nn.Dropout)
assert model.classifier[0].p == 0.1
assert isinstance(model.classifier[1], torch.nn.Linear)
assert model.classifier[1].in_features == 64 * 6 * 22
assert model.classifier[1].out_features == 4 * len(alphabet)
logits = model(images)
assert logits.shape == (1, 4, len(alphabet))

# 在支持 MPS 的设备上执行真实前向传播，确保池化层和模型结构
# 与实际训练设备兼容，避免训练开始后才发现后端支持问题。
if torch.backends.mps.is_available():
    mps_model = SimpleCaptchaCNN().to("mps")
    mps_logits = mps_model(images.to("mps"))
    assert mps_logits.shape == (1, 4, len(alphabet))

# 检查损失可以计算并完成一次反向传播。
criterion = torch.nn.CrossEntropyLoss()
loss = calculate_loss(logits, targets, criterion)
assert torch.isfinite(loss)
loss.backward()

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
    predicted_logits,
    expected.reshape(2, -1),
    num_char=4,
    num_class=len(alphabet),
)
assert accuracy_counts.exact_match_accuracy == 0.5
assert accuracy_counts.character_accuracy == 0.75

print("冒烟测试通过")
print(f"图片形状：{tuple(images.shape)}")
print(f"模型输出形状：{tuple(logits.shape)}")
print(f"初始 loss：{loss.item():.4f}")
