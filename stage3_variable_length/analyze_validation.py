"""分析第三阶段最佳模型的验证集序列错误。

运行方式：
    uv run python -m stage3_variable_length.analyze_validation
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .data import ManifestCaptchaDataset
from .evaluate_checkpoint import load_checkpoint_model
from .metrics import evaluate_predictions
from .train import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_VALIDATION_DIR,
    SUPPORTED_DECODERS,
    choose_device,
    evaluate_model,
)

DEFAULT_REPORT_DIRECTORY = Path("stage3_variable_length/evaluation_reports")


@dataclass(frozen=True)
class EditOperation:
    """目标字符串与预测字符串的一步对齐结果。"""

    kind: str
    expected_index: int | None
    predicted_index: int | None
    expected_character: str
    predicted_character: str


def align_sequences(expected: str, predicted: str) -> list[EditOperation]:
    """返回一条最短 Levenshtein 路径上的字符级对齐操作。"""
    distances = [[0] * (len(predicted) + 1) for _ in range(len(expected) + 1)]
    for expected_index in range(len(expected) + 1):
        distances[expected_index][0] = expected_index
    for predicted_index in range(len(predicted) + 1):
        distances[0][predicted_index] = predicted_index

    for expected_index in range(1, len(expected) + 1):
        for predicted_index in range(1, len(predicted) + 1):
            substitution_cost = int(
                expected[expected_index - 1] != predicted[predicted_index - 1]
            )
            distances[expected_index][predicted_index] = min(
                distances[expected_index - 1][predicted_index] + 1,
                distances[expected_index][predicted_index - 1] + 1,
                distances[expected_index - 1][predicted_index - 1] + substitution_cost,
            )

    operations: list[EditOperation] = []
    expected_index = len(expected)
    predicted_index = len(predicted)
    while expected_index > 0 or predicted_index > 0:
        if (
            expected_index > 0
            and predicted_index > 0
            and expected[expected_index - 1] == predicted[predicted_index - 1]
            and distances[expected_index][predicted_index]
            == distances[expected_index - 1][predicted_index - 1]
        ):
            operations.append(
                EditOperation(
                    kind="match",
                    expected_index=expected_index - 1,
                    predicted_index=predicted_index - 1,
                    expected_character=expected[expected_index - 1],
                    predicted_character=predicted[predicted_index - 1],
                )
            )
            expected_index -= 1
            predicted_index -= 1
        elif (
            expected_index > 0
            and predicted_index > 0
            and distances[expected_index][predicted_index]
            == distances[expected_index - 1][predicted_index - 1] + 1
        ):
            operations.append(
                EditOperation(
                    kind="substitution",
                    expected_index=expected_index - 1,
                    predicted_index=predicted_index - 1,
                    expected_character=expected[expected_index - 1],
                    predicted_character=predicted[predicted_index - 1],
                )
            )
            expected_index -= 1
            predicted_index -= 1
        elif (
            expected_index > 0
            and distances[expected_index][predicted_index]
            == distances[expected_index - 1][predicted_index] + 1
        ):
            operations.append(
                EditOperation(
                    kind="deletion",
                    expected_index=expected_index - 1,
                    predicted_index=None,
                    expected_character=expected[expected_index - 1],
                    predicted_character="",
                )
            )
            expected_index -= 1
        else:
            operations.append(
                EditOperation(
                    kind="insertion",
                    expected_index=None,
                    predicted_index=predicted_index - 1,
                    expected_character="",
                    predicted_character=predicted[predicted_index - 1],
                )
            )
            predicted_index -= 1

    operations.reverse()
    return operations


def _has_adjacent_repeat(label: str) -> bool:
    return any(left == right for left, right in zip(label, label[1:], strict=False))


def _operation_description(operation: EditOperation) -> str:
    return (
        f"{operation.kind}:"
        f"{operation.expected_character}>{operation.predicted_character}"
        f"@{operation.expected_index},{operation.predicted_index}"
    )


def build_validation_analysis(
    records: Sequence[dict[str, Any]],
    expected_labels: Sequence[str],
    predicted_labels: Sequence[str],
    *,
    checkpoint_epoch: int,
    validation_loss: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """把验证预测转换成可序列化汇总和逐错误记录。"""
    if not (len(records) == len(expected_labels) == len(predicted_labels)):
        raise ValueError("records、目标和预测数量必须一致")

    evaluation = evaluate_predictions(expected_labels, predicted_labels)
    operation_counts = Counter({"substitution": 0, "deletion": 0, "insertion": 0})
    primary_error_type_counts: Counter[str] = Counter()
    replacement_pairs: Counter[str] = Counter()
    deleted_characters: Counter[str] = Counter()
    inserted_characters: Counter[str] = Counter()
    length_statistics: dict[int, dict[str, int]] = {}
    adjacent_repeat_total = 0
    adjacent_repeat_correct = 0
    errors: list[dict[str, Any]] = []

    for record, expected, predicted in zip(
        records,
        expected_labels,
        predicted_labels,
        strict=True,
    ):
        operations = align_sequences(expected, predicted)
        edit_operations = [
            operation for operation in operations if operation.kind != "match"
        ]
        edit_distance = len(edit_operations)
        is_correct = expected == predicted
        target_length = len(expected)
        length_statistics.setdefault(
            target_length,
            {"total": 0, "correct": 0, "errors": 0, "edit_distance": 0},
        )
        length_statistics[target_length]["total"] += 1
        length_statistics[target_length]["correct"] += int(is_correct)
        length_statistics[target_length]["errors"] += int(not is_correct)
        length_statistics[target_length]["edit_distance"] += edit_distance

        has_adjacent_repeat = _has_adjacent_repeat(expected)
        if has_adjacent_repeat:
            adjacent_repeat_total += 1
            adjacent_repeat_correct += int(is_correct)

        if is_correct:
            continue

        kinds = {operation.kind for operation in edit_operations}
        primary_error_type = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        primary_error_type_counts[primary_error_type] += 1
        for operation in edit_operations:
            operation_counts[operation.kind] += 1
            if operation.kind == "substitution":
                replacement_pairs[
                    f"{operation.expected_character}->{operation.predicted_character}"
                ] += 1
            elif operation.kind == "deletion":
                deleted_characters[operation.expected_character] += 1
            elif operation.kind == "insertion":
                inserted_characters[operation.predicted_character] += 1

        errors.append(
            {
                "image": str(record.get("image_path", record.get("file", ""))),
                "expected": expected,
                "predicted": predicted,
                "expected_length": len(expected),
                "predicted_length": len(predicted),
                "edit_distance": edit_distance,
                "primary_error_type": primary_error_type,
                "substitution_count": sum(
                    operation.kind == "substitution" for operation in edit_operations
                ),
                "deletion_count": sum(
                    operation.kind == "deletion" for operation in edit_operations
                ),
                "insertion_count": sum(
                    operation.kind == "insertion" for operation in edit_operations
                ),
                "has_adjacent_repeat": has_adjacent_repeat,
                "operations": ";".join(
                    _operation_description(operation) for operation in edit_operations
                ),
            }
        )

    by_length = {}
    for length, statistics in sorted(length_statistics.items()):
        total = statistics["total"]
        error_count = statistics["errors"]
        edit_distance = statistics["edit_distance"]
        by_length[str(length)] = {
            **statistics,
            "accuracy": statistics["correct"] / total,
            "average_edit_distance": edit_distance / total,
            "average_error_edit_distance": (
                edit_distance / error_count if error_count else 0.0
            ),
        }

    summary = {
        "checkpoint_epoch": checkpoint_epoch,
        "validation_loss": validation_loss,
        "total_sequences": evaluation.total_sequences,
        "correct_sequences": evaluation.correct_sequences,
        "error_sequences": (evaluation.total_sequences - evaluation.correct_sequences),
        "exact_match_accuracy": evaluation.exact_match_accuracy,
        "total_edit_distance": evaluation.total_edit_distance,
        "total_target_characters": evaluation.total_target_characters,
        "character_error_rate": evaluation.character_error_rate,
        "operation_counts": dict(operation_counts),
        "primary_error_type_counts": dict(primary_error_type_counts),
        "replacement_pairs": dict(replacement_pairs.most_common()),
        "deleted_characters": dict(deleted_characters.most_common()),
        "inserted_characters": dict(inserted_characters.most_common()),
        "adjacent_repeat": {
            "total": adjacent_repeat_total,
            "correct": adjacent_repeat_correct,
            "errors": adjacent_repeat_total - adjacent_repeat_correct,
            "accuracy": (
                adjacent_repeat_correct / adjacent_repeat_total
                if adjacent_repeat_total
                else 0.0
            ),
        },
        "by_length": by_length,
    }
    return summary, errors


def write_validation_reports(
    summary: dict[str, Any],
    errors: Sequence[dict[str, Any]],
    report_directory: Path,
) -> tuple[Path, Path]:
    """写出验证错误汇总 JSON 和逐错误 CSV。"""
    report_directory.mkdir(parents=True, exist_ok=True)
    summary_path = report_directory / "validation_summary.json"
    errors_path = report_directory / "validation_errors.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    field_names = [
        "image",
        "expected",
        "predicted",
        "expected_length",
        "predicted_length",
        "edit_distance",
        "primary_error_type",
        "substitution_count",
        "deletion_count",
        "insertion_count",
        "has_adjacent_repeat",
        "operations",
    ]
    with errors_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(errors)
    return summary_path, errors_path


def analyze_validation_checkpoint(
    *,
    model_path: Path,
    validation_dir: Path,
    report_directory: Path,
    batch_size: int,
    device_name: str,
    decoder: str = "greedy",
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    """加载最佳模型、只分析 validation manifest 并写出报告。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if decoder not in SUPPORTED_DECODERS:
        supported = ", ".join(SUPPORTED_DECODERS)
        raise ValueError(f"decoder 必须是以下值之一：{supported}")
    if beam_width <= 0:
        raise ValueError("beam_width 必须大于 0")

    device = choose_device(device_name)
    model, checkpoint = load_checkpoint_model(model_path, device)
    dataset = ManifestCaptchaDataset(validation_dir)
    if dataset.split != "validation":
        raise ValueError("只允许分析 split=validation 的数据清单")
    if dataset.characters != model.characters:
        raise ValueError("验证集字符表与 checkpoint 不一致")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loss, _metrics, comparisons = evaluate_model(
        model,
        data_loader,
        device,
        decoder=decoder,
        beam_width=beam_width,
    )
    records = [
        {
            **record,
            "image_path": str(validation_dir / str(record["file"])),
        }
        for record in dataset.records
    ]
    expected_labels = [expected for expected, _predicted in comparisons]
    predicted_labels = [predicted for _expected, predicted in comparisons]
    summary, errors = build_validation_analysis(
        records,
        expected_labels,
        predicted_labels,
        checkpoint_epoch=int(checkpoint.get("epoch", 0)),
        validation_loss=validation_loss,
    )
    summary["checkpoint_path"] = str(model_path)
    summary["checkpoint_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary["decoder"] = decoder
    summary["beam_width"] = beam_width if decoder == "beam" else None
    summary["validation_directory"] = str(validation_dir)
    summary_path, errors_path = write_validation_reports(
        summary,
        errors,
        report_directory,
    )
    return summary, errors, summary_path, errors_path


def _format_counter(values: dict[str, int], limit: int = 10) -> str:
    items = list(values.items())[:limit]
    return ", ".join(f"{key}={value}" for key, value in items) or "无"


def print_analysis(
    summary: dict[str, Any],
    summary_path: Path,
    errors_path: Path,
) -> None:
    """打印最重要的验证错误结论和报告路径。"""
    decoder_description = str(summary.get("decoder", "greedy"))
    if summary.get("decoder") == "beam":
        decoder_description += f"(width={summary.get('beam_width')})"
    print(f"解码器：{decoder_description}")
    print(
        f"验证样本：{summary['total_sequences']} | "
        f"正确：{summary['correct_sequences']} | "
        f"错误：{summary['error_sequences']} | "
        f"整串准确率：{summary['exact_match_accuracy']:.2%} | "
        f"CER：{summary['character_error_rate']:.2%}"
    )
    print(f"字符级操作：{_format_counter(summary['operation_counts'])}")
    print(f"逐字符串主要错误：{_format_counter(summary['primary_error_type_counts'])}")
    repeat = summary["adjacent_repeat"]
    print(
        f"连续重复字符：{repeat['correct']}/{repeat['total']} 正确 "
        f"({repeat['accuracy']:.2%})"
    )
    print("按目标长度：")
    for length, statistics in summary["by_length"].items():
        print(
            f"  {length}位：{statistics['correct']}/{statistics['total']} 正确 | "
            f"准确率={statistics['accuracy']:.2%} | "
            f"编辑距离={statistics['edit_distance']}"
        )
    print(f"替换对：{_format_counter(summary['replacement_pairs'])}")
    print(f"删除字符：{_format_counter(summary['deleted_characters'])}")
    print(f"插入字符：{_format_counter(summary['inserted_characters'])}")
    print(f"汇总报告：{summary_path}")
    print(f"错误明细：{errors_path}")
    print("测试集未读取。")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析第三阶段最佳 checkpoint 的验证集错误",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--decoder",
        choices=SUPPORTED_DECODERS,
        default="greedy",
    )
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    summary, _errors, summary_path, errors_path = analyze_validation_checkpoint(
        model_path=args.model_path,
        validation_dir=args.validation_dir,
        report_directory=args.report_directory,
        batch_size=args.batch_size,
        device_name=args.device,
        decoder=args.decoder,
        beam_width=args.beam_width,
    )
    print_analysis(summary, summary_path, errors_path)


if __name__ == "__main__":
    main()
