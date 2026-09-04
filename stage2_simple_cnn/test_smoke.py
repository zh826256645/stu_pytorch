"""简单 CNN 的最小冒烟测试。

运行方式：
    uv run python -m stage2_simple_cnn.test_smoke
"""

import torch
from torchvision.transforms import ToTensor

from datasets import CaptchaData, alphabet

from .models import SimpleCaptchaCNN
from .train import calculate_loss

# 读取一张真实验证码，检查项目数据能否正常送入模型。
image, target = CaptchaData("./data/test", transform=ToTensor())[0]
assert image.shape == (3, 100, 180)
assert target.shape == (4 * len(alphabet),)

# 增加一个批量维度：[3, 100, 180] -> [1, 3, 100, 180]。
images = image.unsqueeze(0)
targets = target.unsqueeze(0)

model = SimpleCaptchaCNN()
assert isinstance(model.classifier[0], torch.nn.Dropout)
assert model.classifier[0].p == 0.1
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

print("冒烟测试通过")
print(f"图片形状：{tuple(images.shape)}")
print(f"模型输出形状：{tuple(logits.shape)}")
print(f"初始 loss：{loss.item():.4f}")
