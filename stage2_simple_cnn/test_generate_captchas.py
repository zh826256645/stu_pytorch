"""验证码生成器的最小可运行检查。

运行方式：
    uv run python -m stage2_simple_cnn.test_generate_captchas
"""

import json
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .generate_captchas import (
    DEFAULT_CHARACTERS,
    discover_font_paths,
    font_supports_characters,
    generate_balanced_labels,
    generate_dataset,
)

labels = generate_balanced_labels(
    40,
    "abcd",
    random.Random(7),
)
assert len(labels) == 40
assert len(set(labels)) == 40
for position in range(4):
    counts = Counter(label[position] for label in labels)
    assert set(counts) == set("abcd")
    assert max(counts.values()) - min(counts.values()) <= 1

font_paths = discover_font_paths(None, characters=DEFAULT_CHARACTERS)
assert font_paths
assert all("georgian" not in path.name.lower() for path in font_paths)
missing_latin_font = Path("/System/Library/Fonts/SFGeorgianRounded.ttf")
if missing_latin_font.is_file():
    assert not font_supports_characters(missing_latin_font, DEFAULT_CHARACTERS)

with TemporaryDirectory() as temporary_directory:
    temporary_path = Path(temporary_directory)
    first_output = temporary_path / "first"
    second_output = temporary_path / "second"

    first_records = generate_dataset(
        output_dir=first_output,
        count=8,
        seed=11,
        characters=DEFAULT_CHARACTERS,
        font_paths=font_paths[:3],
    )
    second_records = generate_dataset(
        output_dir=second_output,
        count=8,
        seed=11,
        characters=DEFAULT_CHARACTERS,
        font_paths=font_paths[:3],
    )

    assert [record.label for record in first_records] == [
        record.label for record in second_records
    ]
    assert [record.sha256 for record in first_records] == [
        record.sha256 for record in second_records
    ]

    for record in first_records:
        image_path = first_output / record.file
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.size == (180, 100)
            assert image.mode == "RGB"

    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["seed"] == 11
    assert manifest["count"] == 8
    assert manifest["characters"] == DEFAULT_CHARACTERS
    assert len(manifest["files"]) == 8

    try:
        generate_dataset(
            output_dir=first_output,
            count=2,
            seed=11,
            characters=DEFAULT_CHARACTERS,
            font_paths=font_paths[:1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("已有测试图片时应默认拒绝覆盖")

print("验证码生成器测试通过")
