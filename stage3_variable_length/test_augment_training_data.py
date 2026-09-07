"""训练标签多渲染 V2 数据构建的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_augment_training_data
"""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from stage2_simple_cnn.generate_captchas import discover_font_paths

from .augment_training_data import build_multirender_training_dataset
from .generate_captchas import generate_split_datasets

font_path = discover_font_paths(None, characters="ab")[0]
with TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    source_root = generate_split_datasets(
        output_root=root / "v1",
        characters="ab",
        font_paths=[font_path],
        seed=11,
        min_length=2,
        max_length=6,
        train_per_length=1,
        validation_per_length=1,
        test_per_length=1,
    )
    output_root = build_multirender_training_dataset(
        source_train_dir=source_root / "train",
        validation_dir=source_root / "validation",
        test_dir=source_root / "test",
        output_root=root / "v2",
        font_paths=[font_path],
        seed=12,
        variants_per_label=2,
    )

    source_manifest = json.loads(
        (source_root / "train" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_root / "train" / "manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))

    assert manifest["split"] == "train"
    assert manifest["count"] == 10
    assert manifest["unique_label_count"] == 5
    assert manifest["variants_per_label"] == 2
    assert manifest["per_length"] == {str(length): 2 for length in range(2, 7)}
    assert Counter(record["label"] for record in manifest["files"]) == {
        record["label"]: 2 for record in source_manifest["files"]
    }

    source_sha_by_label = {
        record["label"]: record["sha256"] for record in source_manifest["files"]
    }
    original_variants = [
        record for record in manifest["files"] if record["variant_index"] == 0
    ]
    assert all(
        record["sha256"] == source_sha_by_label[record["label"]]
        for record in original_variants
    )
    assert audit["train_validation_label_overlap_count"] == 0
    assert audit["train_test_label_overlap_count"] == 0
    assert audit["train_validation_sha256_overlap_count"] == 0
    assert audit["train_test_sha256_overlap_count"] == 0
    assert audit["intra_train_duplicate_sha256_count"] == 0

    try:
        build_multirender_training_dataset(
            source_train_dir=source_root / "train",
            validation_dir=source_root / "validation",
            test_dir=source_root / "test",
            output_root=output_root,
            font_paths=[font_path],
            seed=12,
            variants_per_label=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("已有 V2 数据时应默认拒绝覆盖")

print("训练标签多渲染 V2 数据测试通过")
