"""正式可变长度数据划分和落盘的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_generate_dataset
"""

import json
import random
from collections import Counter
from tempfile import TemporaryDirectory

from stage2_simple_cnn.generate_captchas import discover_font_paths

from .generate_captchas import build_label_splits, generate_split_datasets

splits = build_label_splits(
    "ab",
    min_length=2,
    max_length=6,
    train_per_length=1,
    validation_per_length=1,
    test_per_length=1,
    rng=random.Random(7),
)

assert set(splits) == {"train", "validation", "test"}
for labels in splits.values():
    assert Counter(map(len, labels)) == {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

train_labels = set(splits["train"])
validation_labels = set(splits["validation"])
test_labels = set(splits["test"])
assert train_labels.isdisjoint(validation_labels)
assert train_labels.isdisjoint(test_labels)
assert validation_labels.isdisjoint(test_labels)

repeated_splits = build_label_splits(
    "ab",
    min_length=2,
    max_length=6,
    train_per_length=1,
    validation_per_length=1,
    test_per_length=1,
    rng=random.Random(7),
)
assert repeated_splits == splits

font_path = discover_font_paths(None, characters="ab")[0]
with TemporaryDirectory() as temporary_directory:
    output_root = generate_split_datasets(
        output_root=temporary_directory,
        characters="ab",
        font_paths=[font_path],
        seed=11,
        min_length=2,
        max_length=6,
        train_per_length=1,
        validation_per_length=1,
        test_per_length=1,
    )

    for split in ("train", "validation", "test"):
        split_directory = output_root / split
        image_paths = sorted(split_directory.glob("*.png"))
        assert len(image_paths) == 5

        manifest = json.loads(
            (split_directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["split"] == split
        assert manifest["count"] == 5
        assert manifest["per_length"] == {
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
        }
        assert len(manifest["files"]) == 5

    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))
    assert audit["label_overlap_count"] == 0
    assert audit["sha256_overlap_count"] == 0

    try:
        generate_split_datasets(
            output_root=output_root,
            characters="ab",
            font_paths=[font_path],
            seed=11,
            min_length=2,
            max_length=6,
            train_per_length=1,
            validation_per_length=1,
            test_per_length=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("已有正式数据时应默认拒绝覆盖")

print("正式数据分层、落盘和审计测试通过")
