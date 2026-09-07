"""可变长度标签生成器的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_generate_captchas
"""

import random
from collections import Counter

from stage2_simple_cnn.generate_captchas import discover_font_paths

from .generate_captchas import (
    generate_variable_length_labels,
    render_variable_length_captcha,
)
from .models import DEFAULT_CHARACTERS

labels = generate_variable_length_labels(
    12,
    DEFAULT_CHARACTERS,
    min_length=2,
    max_length=6,
    rng=random.Random(0),
)
length_counts = Counter(map(len, labels))

assert len(labels) == 12
assert len(set(labels)) == 12
assert set(length_counts) == {2, 3, 4, 5, 6}
assert max(length_counts.values()) - min(length_counts.values()) <= 1
assert all(set(label) <= set(DEFAULT_CHARACTERS) for label in labels)

repeated_character_labels = generate_variable_length_labels(
    4,
    "ab",
    min_length=2,
    max_length=2,
    rng=random.Random(0),
)
assert set(repeated_character_labels) == {"aa", "ab", "ba", "bb"}

same_seed_labels = generate_variable_length_labels(
    12,
    DEFAULT_CHARACTERS,
    min_length=2,
    max_length=6,
    rng=random.Random(0),
)
assert same_seed_labels == labels

font_path = discover_font_paths(None, characters=DEFAULT_CHARACTERS)[0]
for label in ("aa", "2g7xmy"):
    image = render_variable_length_captcha(
        label,
        font_path,
        random.Random(11),
    )
    repeated_image = render_variable_length_captcha(
        label,
        font_path,
        random.Random(11),
    )
    assert image.size == (180, 100)
    assert image.mode == "RGB"
    assert image.getextrema() != ((255, 255), (255, 255), (255, 255))
    assert image.tobytes() == repeated_image.tobytes()

print(f"可变长度标签与渲染测试通过：{dict(sorted(length_counts.items()))}")
