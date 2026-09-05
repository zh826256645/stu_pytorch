"""简单 CNN 的最小冒烟测试。

运行方式：
    uv run python -m stage2_simple_cnn.test_smoke
"""

import torch
from torchvision.transforms import Compose, RandomAffine, ToTensor

from datasets import CaptchaData, alphabet
from training_metrics import calculate_accuracy_counts

from .models import SimpleCaptchaCNN
from .train import (
    build_image_transform,
    build_lr_scheduler,
    calculate_loss,
    is_better_checkpoint,
    set_random_seed,
)

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

# 相同 seed 应产生相同的 PyTorch 随机序列，负数 seed 会被拒绝。
set_random_seed(7)
first_random_values = torch.rand(4)
set_random_seed(7)
second_random_values = torch.rand(4)
assert torch.equal(first_random_values, second_random_values)
try:
    set_random_seed(-1)
except ValueError:
    pass
else:
    raise AssertionError("负数 seed 应抛出 ValueError")

# 整图准确率优先；准确率相同时，验证 loss 更低的权重应覆盖旧权重。
assert is_better_checkpoint(
    accuracy=0.8,
    loss=0.3,
    best_accuracy=0.7,
    best_loss=0.2,
)
assert is_better_checkpoint(
    accuracy=0.8,
    loss=0.2,
    best_accuracy=0.8,
    best_loss=0.3,
)
assert not is_better_checkpoint(
    accuracy=0.8,
    loss=0.4,
    best_accuracy=0.8,
    best_loss=0.3,
)

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
assert model.feature_shape == (64, 6, 22)
assert isinstance(model.bottleneck, torch.nn.Identity)
assert isinstance(model.classifier, torch.nn.Sequential)
assert isinstance(model.classifier[0], torch.nn.Dropout)
assert model.classifier[0].p == 0.1
assert isinstance(model.classifier[1], torch.nn.Linear)
assert model.classifier[1].in_features == 64 * 6 * 22
assert model.classifier[1].out_features == 4 * len(alphabet)
assert sum(parameter.numel() for parameter in model.parameters()) == 1_240_464
logits = model(images)
assert logits.shape == (1, 4, len(alphabet))

# 1×1 卷积瓶颈只把通道从 64 压缩到 32，空间尺寸仍保持 6×22。
bottleneck_model = SimpleCaptchaCNN(bottleneck_channels=32)
assert isinstance(bottleneck_model.bottleneck, torch.nn.Sequential)
assert isinstance(bottleneck_model.bottleneck[0], torch.nn.Conv2d)
assert bottleneck_model.bottleneck[0].in_channels == 64
assert bottleneck_model.bottleneck[0].out_channels == 32
assert bottleneck_model.bottleneck[0].kernel_size == (1, 1)
assert isinstance(bottleneck_model.bottleneck[1], torch.nn.BatchNorm2d)
assert bottleneck_model.bottleneck[1].num_features == 32
assert isinstance(bottleneck_model.bottleneck[2], torch.nn.ReLU)
assert bottleneck_model.feature_shape == (32, 6, 22)
assert bottleneck_model.classifier[1].in_features == 32 * 6 * 22
assert sum(parameter.numel() for parameter in bottleneck_model.parameters()) == 634_352
bottleneck_logits = bottleneck_model(images)
assert bottleneck_logits.shape == (1, 4, len(alphabet))

# 继续压缩至 16 通道，用于验证全连接层参数压缩的有效边界。
small_bottleneck_model = SimpleCaptchaCNN(bottleneck_channels=16)
assert small_bottleneck_model.feature_shape == (16, 6, 22)
assert small_bottleneck_model.bottleneck[0].out_channels == 16
assert small_bottleneck_model.classifier[1].in_features == 16 * 6 * 22
assert (
    sum(parameter.numel() for parameter in small_bottleneck_model.parameters())
    == 329_152
)
small_bottleneck_logits = small_bottleneck_model(images)
assert small_bottleneck_logits.shape == (1, 4, len(alphabet))

# 8 通道瓶颈用于确定继续压缩分类头时的性能边界。
tiny_bottleneck_model = SimpleCaptchaCNN(bottleneck_channels=8)
assert tiny_bottleneck_model.feature_shape == (8, 6, 22)
assert tiny_bottleneck_model.bottleneck[0].out_channels == 8
assert tiny_bottleneck_model.classifier[1].in_features == 8 * 6 * 22
assert (
    sum(parameter.numel() for parameter in tiny_bottleneck_model.parameters())
    == 176_552
)
tiny_bottleneck_logits = tiny_bottleneck_model(images)
assert tiny_bottleneck_logits.shape == (1, 4, len(alphabet))

# 四卷积块实验保持全连接层输入特征数不变，只增加卷积特征提取深度。
four_block_model = SimpleCaptchaCNN(num_conv_blocks=4)
four_block_convolution_layers = [
    layer for layer in four_block_model.features if isinstance(layer, torch.nn.Conv2d)
]
assert [layer.out_channels for layer in four_block_convolution_layers] == [
    16,
    32,
    64,
    128,
]
four_block_batch_norm_layers = [
    layer
    for layer in four_block_model.features
    if isinstance(layer, torch.nn.BatchNorm2d)
]
assert [layer.num_features for layer in four_block_batch_norm_layers] == [
    16,
    32,
    64,
    128,
]
assert four_block_model.feature_shape == (128, 3, 22)
assert four_block_model.classifier[1].in_features == 128 * 3 * 22
assert (
    sum(parameter.numel() for parameter in four_block_model.parameters()) == 1_314_576
)
four_block_logits = four_block_model(images)
assert four_block_logits.shape == (1, 4, len(alphabet))

# 当前最佳候选结构组合四卷积块和 8 通道瓶颈。
best_candidate_model = SimpleCaptchaCNN(
    num_conv_blocks=4,
    bottleneck_channels=8,
)
assert best_candidate_model.feature_shape == (8, 3, 22)
assert best_candidate_model.classifier[1].in_features == 8 * 3 * 22
assert (
    sum(parameter.numel() for parameter in best_candidate_model.parameters()) == 175_144
)
best_candidate_logits = best_candidate_model(images)
assert best_candidate_logits.shape == (1, 4, len(alphabet))

try:
    SimpleCaptchaCNN(num_conv_blocks=5)
except ValueError:
    pass
else:
    raise AssertionError("不支持的卷积块数量应抛出 ValueError")

for invalid_bottleneck_channels in (0, 65):
    try:
        SimpleCaptchaCNN(bottleneck_channels=invalid_bottleneck_channels)
    except ValueError:
        pass
    else:
        raise AssertionError("无效的瓶颈通道数应抛出 ValueError")

# 在支持 MPS 的设备上执行真实前向传播，确保池化层和模型结构
# 与实际训练设备兼容，避免训练开始后才发现后端支持问题。
if torch.backends.mps.is_available():
    mps_images = images.to("mps")
    mps_model = SimpleCaptchaCNN().to("mps")
    mps_logits = mps_model(mps_images)
    assert mps_logits.shape == (1, 4, len(alphabet))

    bottleneck_mps_model = SimpleCaptchaCNN(bottleneck_channels=32).to("mps")
    bottleneck_mps_logits = bottleneck_mps_model(mps_images)
    assert bottleneck_mps_logits.shape == (1, 4, len(alphabet))

    small_bottleneck_mps_model = SimpleCaptchaCNN(bottleneck_channels=16).to("mps")
    small_bottleneck_mps_logits = small_bottleneck_mps_model(mps_images)
    assert small_bottleneck_mps_logits.shape == (1, 4, len(alphabet))

    tiny_bottleneck_mps_model = SimpleCaptchaCNN(bottleneck_channels=8).to("mps")
    tiny_bottleneck_mps_logits = tiny_bottleneck_mps_model(mps_images)
    assert tiny_bottleneck_mps_logits.shape == (1, 4, len(alphabet))

    four_block_mps_model = SimpleCaptchaCNN(num_conv_blocks=4).to("mps")
    four_block_mps_logits = four_block_mps_model(mps_images)
    assert four_block_mps_logits.shape == (1, 4, len(alphabet))

    best_candidate_mps_model = SimpleCaptchaCNN(
        num_conv_blocks=4,
        bottleneck_channels=8,
    ).to("mps")
    best_candidate_mps_logits = best_candidate_mps_model(mps_images)
    assert best_candidate_mps_logits.shape == (1, 4, len(alphabet))

# 检查损失可以计算并完成一次反向传播。
criterion = torch.nn.CrossEntropyLoss()
loss = calculate_loss(logits, targets, criterion)
assert torch.isfinite(loss)
loss.backward()

bottleneck_loss = calculate_loss(bottleneck_logits, targets, criterion)
assert torch.isfinite(bottleneck_loss)
bottleneck_loss.backward()

small_bottleneck_loss = calculate_loss(small_bottleneck_logits, targets, criterion)
assert torch.isfinite(small_bottleneck_loss)
small_bottleneck_loss.backward()

tiny_bottleneck_loss = calculate_loss(tiny_bottleneck_logits, targets, criterion)
assert torch.isfinite(tiny_bottleneck_loss)
tiny_bottleneck_loss.backward()

best_candidate_loss = calculate_loss(best_candidate_logits, targets, criterion)
assert torch.isfinite(best_candidate_loss)
best_candidate_loss.backward()

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
