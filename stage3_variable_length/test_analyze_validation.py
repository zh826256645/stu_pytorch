"""验证集序列错误对齐与报告的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_analyze_validation
"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .analyze_validation import (
    align_sequences,
    build_validation_analysis,
    parse_args,
    write_validation_reports,
)

beam_args = parse_args(["--decoder", "beam", "--beam-width", "10"])
assert beam_args.decoder == "beam"
assert beam_args.beam_width == 10


def operation_values(expected: str, predicted: str) -> list[tuple[str, str, str]]:
    return [
        (
            operation.kind,
            operation.expected_character,
            operation.predicted_character,
        )
        for operation in align_sequences(expected, predicted)
    ]


assert operation_values("ab", "ac") == [
    ("match", "a", "a"),
    ("substitution", "b", "c"),
]
assert operation_values("ab", "a") == [
    ("match", "a", "a"),
    ("deletion", "b", ""),
]
assert operation_values("a", "ab") == [
    ("match", "a", "a"),
    ("insertion", "", "b"),
]

records = [
    {"file": "substitution.png"},
    {"file": "deletion.png"},
    {"file": "repeat_correct.png"},
    {"file": "insertion.png"},
]
summary, errors = build_validation_analysis(
    records,
    ["ab", "abc", "aa", "b"],
    ["ac", "ab", "aa", "bb"],
    checkpoint_epoch=3,
    validation_loss=0.5,
)
assert summary["total_sequences"] == 4
assert summary["correct_sequences"] == 1
assert summary["error_sequences"] == 3
assert summary["total_edit_distance"] == 3
assert summary["operation_counts"] == {
    "substitution": 1,
    "deletion": 1,
    "insertion": 1,
}
assert summary["replacement_pairs"] == {"b->c": 1}
assert summary["deleted_characters"] == {"c": 1}
assert summary["inserted_characters"] == {"b": 1}
assert summary["adjacent_repeat"]["total"] == 1
assert summary["adjacent_repeat"]["correct"] == 1
assert len(errors) == 3

with TemporaryDirectory() as temporary_directory:
    report_directory = Path(temporary_directory)
    write_validation_reports(summary, errors, report_directory)
    saved_summary = json.loads(
        (report_directory / "validation_summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["operation_counts"]["deletion"] == 1
    with (report_directory / "validation_errors.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        error_rows = list(csv.DictReader(file))
    assert len(error_rows) == 3
    assert {row["primary_error_type"] for row in error_rows} == {
        "substitution",
        "deletion",
        "insertion",
    }

print("验证错误对齐、汇总和报告测试通过")
