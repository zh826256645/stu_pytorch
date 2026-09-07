"""为第三阶段 V1 训练标签增加独立渲染版本。

运行方式：
    uv run python -m stage3_variable_length.augment_training_data
"""

import argparse
import json
import random
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from stage2_simple_cnn.generate_captchas import (
    calculate_sha256,
    discover_font_paths,
)

from .generate_captchas import render_variable_length_captcha

DEFAULT_SOURCE_ROOT = Path("data/stage3_variable_length")
DEFAULT_OUTPUT_ROOT = Path("data/stage3_variable_length_v2")
DEFAULT_SEED = 20250312


def _load_manifest(split_directory: Path, expected_split: str) -> dict[str, object]:
    manifest_path = split_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"没有找到数据清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != expected_split:
        raise ValueError(f"{manifest_path} 的 split 不是 {expected_split}")
    files = manifest.get("files")
    if not isinstance(files, list) or int(manifest.get("count", -1)) != len(files):
        raise ValueError(f"{manifest_path} 的 count 与 files 不一致")
    return manifest


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _manifest_labels(manifest: dict[str, object]) -> set[str]:
    return {str(record["label"]) for record in manifest["files"]}


def _manifest_sha256s(manifest: dict[str, object]) -> set[str]:
    return {str(record["sha256"]) for record in manifest["files"]}


def build_multirender_training_dataset(
    *,
    source_train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
    output_root: Path,
    font_paths: list[Path],
    seed: int,
    variants_per_label: int = 2,
    replace_existing: bool = False,
    show_progress: bool = False,
) -> Path:
    """保留每张 V1 训练图，再为其标签增加独立随机渲染。"""
    if not font_paths:
        raise ValueError("font_paths 不能为空")
    if seed < 0:
        raise ValueError("seed 必须大于等于 0")
    if variants_per_label < 2:
        raise ValueError("variants_per_label 必须至少为 2")

    source_train_dir = Path(source_train_dir)
    validation_dir = Path(validation_dir)
    test_dir = Path(test_dir)
    output_root = Path(output_root)
    source_manifest = _load_manifest(source_train_dir, "train")
    validation_manifest = _load_manifest(validation_dir, "validation")
    test_manifest = _load_manifest(test_dir, "test")

    characters = str(source_manifest["characters"])
    width = int(source_manifest["width"])
    height = int(source_manifest["height"])
    for split, manifest in (
        ("validation", validation_manifest),
        ("test", test_manifest),
    ):
        if str(manifest["characters"]) != characters:
            raise ValueError(f"{split} 字符表与训练集不一致")
        if (int(manifest["width"]), int(manifest["height"])) != (
            width,
            height,
        ):
            raise ValueError(f"{split} 图片尺寸与训练集不一致")

    source_records = source_manifest["files"]
    source_labels = [str(record["label"]) for record in source_records]
    if len(source_labels) != len(set(source_labels)):
        raise ValueError("V1 训练 manifest 的标签必须唯一")

    if output_root.exists() and any(output_root.iterdir()):
        if not replace_existing:
            raise ValueError(f"输出目录 {output_root} 已有内容；如确定重建请启用覆盖")
        shutil.rmtree(output_root)
    train_output_directory = output_root / "train"
    train_output_directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    output_records: list[dict[str, object]] = []
    for label_index, source_record in enumerate(source_records, start=1):
        label = str(source_record["label"])
        source_path = source_train_dir / str(source_record["file"])
        if not source_path.is_file():
            raise ValueError(f"V1 训练图片不存在：{source_path}")

        original_output_path = train_output_directory / f"{label}_v0.png"
        shutil.copy2(source_path, original_output_path)
        output_records.append(
            {
                "file": original_output_path.name,
                "label": label,
                "length": len(label),
                "sha256": calculate_sha256(original_output_path),
                "variant_index": 0,
                "source": "v1_original",
                "source_file": str(source_path),
                "font": source_record.get("font"),
                "render_seed": source_record.get("render_seed"),
            }
        )

        for variant_index in range(1, variants_per_label):
            render_seed = rng.randrange(2**63)
            font_path = rng.choice(font_paths)
            image = render_variable_length_captcha(
                label,
                font_path,
                random.Random(render_seed),
                width=width,
                height=height,
            )
            output_path = train_output_directory / f"{label}_v{variant_index}.png"
            image.save(output_path, format="PNG", optimize=True)
            output_records.append(
                {
                    "file": output_path.name,
                    "label": label,
                    "length": len(label),
                    "sha256": calculate_sha256(output_path),
                    "variant_index": variant_index,
                    "source": "new_render",
                    "source_file": None,
                    "font": str(font_path),
                    "render_seed": render_seed,
                }
            )

        if show_progress and (
            label_index % 100 == 0 or label_index == len(source_records)
        ):
            print(
                f"train: 已处理 {label_index}/{len(source_records)} 个标签，"
                f"生成 {label_index * variants_per_label} 张图片"
            )

    manifest = {
        "version": 1,
        "generator": "stage3_variable_length.augment_training_data",
        "split": "train",
        "seed": seed,
        "source_train_manifest": str(source_train_dir / "manifest.json"),
        "count": len(output_records),
        "unique_label_count": len(source_labels),
        "variants_per_label": variants_per_label,
        "per_length": dict(
            sorted(
                Counter(len(str(record["label"])) for record in output_records).items()
            )
        ),
        "min_length": int(source_manifest["min_length"]),
        "max_length": int(source_manifest["max_length"]),
        "width": width,
        "height": height,
        "characters": characters,
        "fonts": [str(path) for path in font_paths],
        "files": output_records,
    }
    _write_json(train_output_directory / "manifest.json", manifest)

    train_label_set = set(source_labels)
    validation_label_set = _manifest_labels(validation_manifest)
    test_label_set = _manifest_labels(test_manifest)
    train_sha256_values = [str(record["sha256"]) for record in output_records]
    train_sha256_set = set(train_sha256_values)
    validation_sha256_set = _manifest_sha256s(validation_manifest)
    test_sha256_set = _manifest_sha256s(test_manifest)
    audit = {
        "version": 1,
        "seed": seed,
        "train_image_count": len(output_records),
        "train_unique_label_count": len(train_label_set),
        "variants_per_label": variants_per_label,
        "train_validation_label_overlap_count": len(
            train_label_set & validation_label_set
        ),
        "train_test_label_overlap_count": len(train_label_set & test_label_set),
        "train_validation_sha256_overlap_count": len(
            train_sha256_set & validation_sha256_set
        ),
        "train_test_sha256_overlap_count": len(train_sha256_set & test_sha256_set),
        "intra_train_duplicate_sha256_count": (
            len(train_sha256_values) - len(train_sha256_set)
        ),
    }
    _write_json(output_root / "audit.json", audit)

    failed_checks = [
        name
        for name, value in audit.items()
        if name.endswith("_overlap_count") or name.endswith("_sha256_count")
        if int(value) != 0
    ]
    if failed_checks:
        raise ValueError(f"V2 数据审计失败：{failed_checks}")
    return output_root


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="保留 V1 训练图片并为每个训练标签增加独立渲染版本",
    )
    parser.add_argument(
        "--source-train-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "train",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "validation",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "test",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--variants-per-label", type=int, default=2)
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
    source_manifest = _load_manifest(args.source_train_dir, "train")
    font_paths = discover_font_paths(
        args.font_path,
        characters=str(source_manifest["characters"]),
    )
    output_root = build_multirender_training_dataset(
        source_train_dir=args.source_train_dir,
        validation_dir=args.validation_dir,
        test_dir=args.test_dir,
        output_root=args.output_root,
        font_paths=font_paths,
        seed=args.seed,
        variants_per_label=args.variants_per_label,
        replace_existing=args.replace_existing,
        show_progress=True,
    )
    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))
    print(f"V2 训练数据已生成到：{output_root / 'train'}")
    print(
        f"图片：{audit['train_image_count']} | "
        f"唯一标签：{audit['train_unique_label_count']} | "
        f"每标签渲染数：{audit['variants_per_label']}"
    )
    print(
        f"验证标签交集：{audit['train_validation_label_overlap_count']} | "
        f"测试标签交集：{audit['train_test_label_overlap_count']} | "
        f"验证图片 SHA 交集：{audit['train_validation_sha256_overlap_count']} | "
        f"测试图片 SHA 交集：{audit['train_test_sha256_overlap_count']} | "
        f"训练内部重复 SHA：{audit['intra_train_duplicate_sha256_count']}"
    )
    print("测试集只用于 manifest 身份审计，没有运行模型或计算测试指标。")


if __name__ == "__main__":
    main()
