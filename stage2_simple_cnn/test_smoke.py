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
logits = model(images)
assert logits.shape == (1, 4, len(alphabet))

# 检查损失可以计算并完成一次反向传播。
criterion = torch.nn.CrossEntropyLoss()
loss = calculate_loss(logits, targets, criterion)
assert torch.isfinite(loss)
loss.backward()

print("冒烟测试通过")
print(f"图片形状：{tuple(images.shape)}")
print(f"模型输出形状：{tuple(logits.shape)}")
print(f"初始 loss：{loss.item():.4f}")
