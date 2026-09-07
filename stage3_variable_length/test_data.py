"""极小可变长度验证码数据集的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_data
"""

import torch
from torch.utils.data import DataLoader

from .data import TinyCaptchaDataset

labels = ["aa", "a7b", "b8x2p6"]
dataset = TinyCaptchaDataset(labels, seed=0)

assert len(dataset) == len(labels)
image, label = dataset[0]
assert image.shape == (3, 100, 180)
assert image.dtype == torch.float32
assert 0 <= image.min() <= image.max() <= 1
assert label == "aa"
assert torch.equal(image, dataset[0][0])

images, batch_labels = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))
assert images.shape == (3, 3, 100, 180)
assert list(batch_labels) == labels

print(f"极小数据集测试通过：images={tuple(images.shape)}, labels={list(batch_labels)}")
