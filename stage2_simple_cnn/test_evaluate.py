"""独立测试评估与混淆矩阵报告的最小测试。

运行方式：
    uv run python -m stage2_simple_cnn.test_evaluate
"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from datasets import make_dataset

from .evaluate import (
    build_evaluation_result,
    find_exact_duplicates,
    write_reports,
)

class_names = "ab"
expected_indices = torch.tensor([[0, 1], [0, 1]])
predicted_indices = torch.tensor([[0, 1], [1, 1]])
predicted_confidences = torch.tensor([[0.9, 0.8], [0.6, 0.7]])
image_paths = [Path("ab.png"), Path("ab_second.png")]

result = build_evaluation_result(
    predicted_indices,
    expected_indices,
    predicted_confidences,
    image_paths,
    class_names=class_names,
)
assert result.total_captchas == 2
assert result.correct_captchas == 1
assert result.exact_match_accuracy == 0.5
assert result.correct_characters == 3
assert result.total_characters == 4
assert result.character_accuracy == 0.75
assert result.per_position_correct == (1, 2)
assert result.per_position_accuracy == (0.5, 1.0)
assert result.confusion_matrix == ((1, 1), (0, 2))
assert len(result.mistakes) == 1
assert result.mistakes[0].image == "ab_second.png"
assert result.mistakes[0].expected == "ab"
assert result.mistakes[0].predicted == "bb"
assert result.mistakes[0].wrong_positions == (1,)

with TemporaryDirectory() as temporary_directory:
    temporary_path = Path(temporary_directory)
    test_dir = temporary_path / "test"
    reference_dir = temporary_path / "reference"
    test_dir.mkdir()
    reference_dir.mkdir()

    (test_dir / "ab.png").write_bytes(b"same image")
    (test_dir / "ba.png").write_bytes(b"new image")
    (test_dir / ".DS_Store").write_bytes(b"ignored metadata")
    (reference_dir / "ab.png").write_bytes(b"same image")

    samples = make_dataset(
        str(test_dir),
        alphabet=class_names,
        num_class=len(class_names),
        num_char=2,
    )
    assert len(samples) == 2

    duplicates = find_exact_duplicates(test_dir, [reference_dir])
    assert len(duplicates) == 1
    assert Path(duplicates[0].test_image).name == "ab.png"
    assert Path(duplicates[0].reference_image).name == "ab.png"

    report_dir = temporary_path / "reports"
    write_reports(
        result,
        report_dir,
        duplicates,
        class_names=class_names,
    )
    assert (report_dir / "metrics.json").is_file()
    assert (report_dir / "confusion_matrix.csv").is_file()
    assert (report_dir / "confusion_matrix.png").is_file()
    assert (report_dir / "errors.csv").is_file()

    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["character_accuracy"] == 0.75
    assert metrics["error_count"] == 1
    assert metrics["exact_duplicate_count"] == 1

    with (report_dir / "errors.csv").open(encoding="utf-8", newline="") as file:
        error_rows = list(csv.DictReader(file))
    assert len(error_rows) == 1
    assert error_rows[0]["expected"] == "ab"
    assert error_rows[0]["predicted"] == "bb"
    assert error_rows[0]["wrong_positions"] == "1"

print("独立测试评估测试通过")
