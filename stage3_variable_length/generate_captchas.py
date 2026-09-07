"""第三阶段可变长度验证码数据生成工具。"""

import argparse
import json
import random
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageFilter

from stage2_simple_cnn.generate_captchas import (
    FOREGROUND_COLORS,
    calculate_sha256,
    create_background,
    discover_font_paths,
    draw_interference_lines,
    render_character,
)

from .models import DEFAULT_CHARACTERS

MEDIUM_FOREGROUND_COLORS = tuple(
    color for color in FOREGROUND_COLORS if color != (245, 220, 0)
)


def _label_from_number(number: int, length: int, characters: str) -> str:
    """把整数转换成固定长度的字符表进制标签。"""
    base = len(characters)
    result = [characters[0]] * length
    for position in range(length - 1, -1, -1):
        number, character_index = divmod(number, base)
        result[position] = characters[character_index]
    return "".join(result)


def generate_variable_length_labels(
    count: int,
    characters: str,
    *,
    min_length: int,
    max_length: int,
    rng: random.Random,
) -> list[str]:
    """生成唯一标签，并让各个字符长度的样本数尽可能均衡。"""
    if count <= 0:
        raise ValueError("count 必须大于 0")
    if len(characters) < 2 or len(set(characters)) != len(characters):
        raise ValueError("characters 至少需要两个不重复字符")
    if min_length <= 0 or max_length < min_length:
        raise ValueError("字符长度范围无效")

    lengths = list(range(min_length, max_length + 1))
    samples_per_length, remainder = divmod(count, len(lengths))
    counts = {length: samples_per_length for length in lengths}

    extra_lengths = lengths.copy()
    rng.shuffle(extra_lengths)
    for length in extra_lengths[:remainder]:
        counts[length] += 1

    labels: list[str] = []
    for length in lengths:
        capacity = len(characters) ** length
        required = counts[length]
        if required > capacity:
            raise ValueError(
                f"长度 {length} 只能生成 {capacity} 个唯一标签，"
                f"但当前分配了 {required} 个"
            )

        selected_numbers = rng.sample(range(capacity), required)
        labels.extend(
            _label_from_number(number, length, characters)
            for number in selected_numbers
        )

    rng.shuffle(labels)
    return labels


def build_label_splits(
    characters: str,
    *,
    min_length: int,
    max_length: int,
    train_per_length: int,
    validation_per_length: int,
    test_per_length: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    """按长度分层生成三个集合，并保证所有标签全局唯一。"""
    split_counts = {
        "train": train_per_length,
        "validation": validation_per_length,
        "test": test_per_length,
    }
    if any(count < 0 for count in split_counts.values()):
        raise ValueError("每个集合的样本数不能小于 0")
    if sum(split_counts.values()) <= 0:
        raise ValueError("每种长度至少需要生成一个样本")
    if min_length <= 0 or max_length < min_length:
        raise ValueError("字符长度范围无效")

    labels_by_split = {split: [] for split in split_counts}
    required_per_length = sum(split_counts.values())
    for length in range(min_length, max_length + 1):
        capacity = len(characters) ** length
        if required_per_length > capacity:
            raise ValueError(
                f"长度 {length} 只能生成 {capacity} 个唯一标签，"
                f"但三个集合共需要 {required_per_length} 个"
            )

        selected_numbers = rng.sample(range(capacity), required_per_length)
        labels_for_length = [
            _label_from_number(number, length, characters)
            for number in selected_numbers
        ]
        start = 0
        for split, count in split_counts.items():
            end = start + count
            labels_by_split[split].extend(labels_for_length[start:end])
            start = end

    for labels in labels_by_split.values():
        rng.shuffle(labels)
    return labels_by_split


def render_variable_length_captcha(
    label: str,
    font_path: Path,
    rng: random.Random,
    *,
    width: int = 180,
    height: int = 100,
) -> Image.Image:
    """按标签长度动态布局并渲染一张中等难度验证码。"""
    if not 2 <= len(label) <= 6:
        raise ValueError("label 长度必须在 2 到 6 之间")
    if width < 80 or height < 40:
        raise ValueError("图片尺寸过小")

    image, _background_style = create_background(width, height, rng)
    foreground = rng.choice(MEDIUM_FOREGROUND_COLORS)
    draw_interference_lines(
        image,
        rng,
        foreground,
        count=rng.choice((0, 0, 1)),
    )

    slot_width = width / len(label)
    base_font_size = max(40, 76 - len(label) * 6)
    horizontal_jitter = min(4.0, slot_width * 0.12)
    maximum_layer_width = max(1, round(slot_width * 1.15))

    for position, character in enumerate(label):
        layer = render_character(
            character,
            font_path,
            base_font_size + rng.randint(-2, 2),
            foreground,
            rng,
        )
        if layer.width > maximum_layer_width:
            scale = maximum_layer_width / layer.width
            layer = layer.resize(
                (
                    maximum_layer_width,
                    max(1, round(layer.height * scale)),
                ),
                Image.Resampling.BICUBIC,
            )

        center_x = (position + 0.5) * slot_width
        center_x += rng.uniform(-horizontal_jitter, horizontal_jitter)
        x = round(center_x - layer.width / 2)
        x = max(0, min(width - layer.width, x))

        centered_y = (height - layer.height) / 2
        y = round(centered_y + rng.uniform(-6, 6))
        y = max(0, min(height - layer.height, y))
        image.paste(layer, (x, y), layer)

    draw_interference_lines(
        image,
        rng,
        foreground,
        count=rng.choice((0, 0, 1)),
    )
    blur_radius = rng.choice((0.0, 0.0, 0.0, 0.25))
    if blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image.convert("RGB")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_split_datasets(
    *,
    output_root: Path | str,
    characters: str,
    font_paths: list[Path],
    seed: int,
    min_length: int,
    max_length: int,
    train_per_length: int,
    validation_per_length: int,
    test_per_length: int,
    width: int = 180,
    height: int = 100,
    replace_existing: bool = False,
    show_progress: bool = False,
) -> Path:
    """生成训练、验证和测试图片，写入清单并审计集合间重复。"""
    if not font_paths:
        raise ValueError("font_paths 不能为空")
    if seed < 0:
        raise ValueError("seed 必须大于等于 0")

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not replace_existing:
            raise ValueError(f"输出目录 {output_root} 已有内容；如确定重建请启用覆盖")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    labels_by_split = build_label_splits(
        characters,
        min_length=min_length,
        max_length=max_length,
        train_per_length=train_per_length,
        validation_per_length=validation_per_length,
        test_per_length=test_per_length,
        rng=rng,
    )

    records_by_split: dict[str, list[dict[str, object]]] = {}
    for split, labels in labels_by_split.items():
        split_directory = output_root / split
        split_directory.mkdir()
        records: list[dict[str, object]] = []

        for index, label in enumerate(labels, start=1):
            render_seed = rng.randrange(2**63)
            font_path = rng.choice(font_paths)
            image = render_variable_length_captcha(
                label,
                font_path,
                random.Random(render_seed),
                width=width,
                height=height,
            )
            output_path = split_directory / f"{label}.png"
            image.save(output_path, format="PNG", optimize=True)
            records.append(
                {
                    "file": output_path.name,
                    "label": label,
                    "length": len(label),
                    "sha256": calculate_sha256(output_path),
                    "font": str(font_path),
                    "render_seed": render_seed,
                }
            )
            if show_progress and (index % 100 == 0 or index == len(labels)):
                print(f"{split}: 已生成 {index}/{len(labels)} 张")

        records_by_split[split] = records
        manifest = {
            "version": 1,
            "generator": "stage3_variable_length.generate_captchas",
            "split": split,
            "seed": seed,
            "count": len(records),
            "per_length": dict(sorted(Counter(map(len, labels)).items())),
            "min_length": min_length,
            "max_length": max_length,
            "width": width,
            "height": height,
            "characters": characters,
            "fonts": [str(path) for path in font_paths],
            "files": records,
        }
        _write_json(split_directory / "manifest.json", manifest)

    split_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    label_sets = {
        split: {str(record["label"]) for record in records}
        for split, records in records_by_split.items()
    }
    sha256_sets = {
        split: {str(record["sha256"]) for record in records}
        for split, records in records_by_split.items()
    }
    label_overlaps = {
        f"{left}_{right}": sorted(label_sets[left] & label_sets[right])
        for left, right in split_pairs
    }
    sha256_overlaps = {
        f"{left}_{right}": sorted(sha256_sets[left] & sha256_sets[right])
        for left, right in split_pairs
    }
    audit = {
        "version": 1,
        "seed": seed,
        "split_counts": {
            split: len(records) for split, records in records_by_split.items()
        },
        "label_overlap_count": sum(len(values) for values in label_overlaps.values()),
        "sha256_overlap_count": sum(len(values) for values in sha256_overlaps.values()),
        "label_overlaps": label_overlaps,
        "sha256_overlaps": sha256_overlaps,
    }
    _write_json(output_root / "audit.json", audit)
    return output_root


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成第三阶段 2～6 位正式合成训练、验证和测试数据",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/stage3_variable_length"),
    )
    parser.add_argument("--seed", type=int, default=20250311)
    parser.add_argument("--characters", default=DEFAULT_CHARACTERS)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=6)
    parser.add_argument("--train-per-length", type=int, default=300)
    parser.add_argument("--validation-per-length", type=int, default=50)
    parser.add_argument("--test-per-length", type=int, default=50)
    parser.add_argument("--width", type=int, default=180)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument(
        "--font-path",
        type=Path,
        action="append",
        default=None,
        help="字体文件或目录；可重复传入，默认扫描系统字体",
    )
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    font_paths = discover_font_paths(
        args.font_path,
        characters=args.characters,
    )
    output_root = generate_split_datasets(
        output_root=args.output_root,
        characters=args.characters,
        font_paths=font_paths,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
        train_per_length=args.train_per_length,
        validation_per_length=args.validation_per_length,
        test_per_length=args.test_per_length,
        width=args.width,
        height=args.height,
        replace_existing=args.replace_existing,
        show_progress=True,
    )
    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))
    print(f"数据已生成到：{output_root}")
    print(
        f"标签交集：{audit['label_overlap_count']} | "
        f"图片 SHA-256 交集：{audit['sha256_overlap_count']}"
    )


if __name__ == "__main__":
    main()
