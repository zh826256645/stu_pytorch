"""基于 manifest 的正式验证码 Dataset 行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_manifest_data
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import ManifestCaptchaDataset

with TemporaryDirectory() as temporary_directory:
    split_directory = Path(temporary_directory)
    Image.new("RGB", (180, 100), "white").save(split_directory / "first.png")
    Image.new("RGB", (180, 100), "black").save(split_directory / "second.png")
    manifest = {
        "version": 1,
        "split": "train",
        "count": 2,
        "width": 180,
        "height": 100,
        "characters": "ab",
        "files": [
            {"file": "first.png", "label": "aa", "length": 2},
            {"file": "second.png", "label": "aba", "length": 3},
        ],
    }
    (split_directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    dataset = ManifestCaptchaDataset(split_directory)
    assert len(dataset) == 2
    assert dataset.split == "train"
    assert dataset.characters == "ab"

    image, label = dataset[1]
    assert image.shape == (3, 100, 180)
    assert image.dtype == torch.float32
    assert image.min() == 0
    assert image.max() == 0
    assert label == "aba"

    images, labels = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert images.shape == (2, 3, 100, 180)
    assert list(labels) == ["aa", "aba"]

print("manifest Dataset 测试通过")
