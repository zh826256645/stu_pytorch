"""第三阶段用于训练链路诊断的极小内存数据集。"""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .models import DEFAULT_CHARACTERS

DEFAULT_WIDTH = 180
DEFAULT_HEIGHT = 100
MIN_LABEL_LENGTH = 2
MAX_LABEL_LENGTH = 6


class ManifestCaptchaDataset(Dataset):
    """根据生成清单读取正式可变长度验证码图片和字符串标签。"""

    def __init__(self, split_directory: Path | str) -> None:
        super().__init__()
        self.split_directory = Path(split_directory)
        manifest_path = self.split_directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"没有找到数据清单：{manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.split = str(manifest["split"])
        self.characters = str(manifest["characters"])
        self.width = int(manifest["width"])
        self.height = int(manifest["height"])
        self.records = list(manifest["files"])

        if int(manifest["count"]) != len(self.records):
            raise ValueError("manifest 的 count 与 files 数量不一致")
        for record in self.records:
            image_path = self.split_directory / str(record["file"])
            if not image_path.is_file():
                raise ValueError(f"manifest 中的图片不存在：{image_path}")
            label = str(record["label"])
            if int(record["length"]) != len(label):
                raise ValueError(f"标签长度记录不一致：{image_path.name}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        image_path = self.split_directory / str(record["file"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (self.width, self.height):
                raise ValueError(
                    f"图片尺寸应为 {(self.width, self.height)}，"
                    f"实际为 {image.size}：{image_path.name}"
                )
            image_tensor = pil_to_tensor(image).to(dtype=torch.float32).div(255)
        return image_tensor, str(record["label"])


class TinyCaptchaDataset(Dataset):
    """一次性生成确定性的干净验证码，用于检查模型能否过拟合。"""

    def __init__(
        self,
        labels: Sequence[str],
        *,
        seed: int = 0,
        characters: str = DEFAULT_CHARACTERS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        super().__init__()
        if not labels:
            raise ValueError("labels 不能为空")

        character_set = set(characters)
        for label in labels:
            if not MIN_LABEL_LENGTH <= len(label) <= MAX_LABEL_LENGTH:
                raise ValueError("标签长度必须在 2 到 6 之间")
            if not set(label) <= character_set:
                raise ValueError(f"标签包含字符表之外的字符：{label!r}")

        self.labels = list(labels)
        self.images = [
            self._render_clean_captcha(
                label,
                random.Random(seed + index),
                width,
                height,
            )
            for index, label in enumerate(self.labels)
        ]

    @staticmethod
    def _render_clean_captcha(
        label: str,
        rng: random.Random,
        width: int,
        height: int,
    ) -> torch.Tensor:
        background = rng.randint(235, 250)
        image = Image.new("RGB", (width, height), (background,) * 3)
        draw = ImageDraw.Draw(image)

        font_size = min(58, max(28, int((width - 24) / (len(label) * 0.65))))
        font = ImageFont.load_default(size=font_size)
        bounding_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        text_width = bounding_box[2] - bounding_box[0]
        text_height = bounding_box[3] - bounding_box[1]
        x = (width - text_width) // 2 - bounding_box[0] + rng.randint(-5, 5)
        y = (height - text_height) // 2 - bounding_box[1] + rng.randint(-4, 4)
        foreground = rng.randint(15, 65)

        draw.text(
            (x, y),
            label,
            font=font,
            fill=(foreground,) * 3,
            stroke_width=1,
            stroke_fill=(foreground,) * 3,
        )
        return pil_to_tensor(image).to(dtype=torch.float32).div(255)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return self.images[index], self.labels[index]
