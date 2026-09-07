"""第三阶段可变长度识别的评价指标。"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LengthEvaluation:
    """一个目标字符串长度的整串识别指标。"""

    length: int
    total: int
    correct: int
    accuracy: float


@dataclass(frozen=True)
class SequenceEvaluation:
    """一批可变长度预测的总体和分长度指标。"""

    total_sequences: int
    correct_sequences: int
    exact_match_accuracy: float
    total_edit_distance: int
    total_target_characters: int
    character_error_rate: float
    per_length: dict[int, LengthEvaluation]


def levenshtein_distance(expected: str, predicted: str) -> int:
    """返回把 expected 变成 predicted 所需的最少单字符操作数。"""
    previous_row = list(range(len(predicted) + 1))

    for expected_index, expected_character in enumerate(expected, start=1):
        current_row = [expected_index]
        for predicted_index, predicted_character in enumerate(predicted, start=1):
            deletion_cost = previous_row[predicted_index] + 1
            insertion_cost = current_row[predicted_index - 1] + 1
            substitution_cost = previous_row[predicted_index - 1]
            if expected_character != predicted_character:
                substitution_cost += 1
            current_row.append(min(deletion_cost, insertion_cost, substitution_cost))
        previous_row = current_row

    return previous_row[-1]


def evaluate_predictions(
    expected_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> SequenceEvaluation:
    """汇总整串准确率、字符错误率和按目标长度准确率。"""
    if not expected_labels:
        raise ValueError("expected_labels 不能为空")
    if len(expected_labels) != len(predicted_labels):
        raise ValueError("目标标签数与预测标签数必须相同")
    if any(not label for label in expected_labels):
        raise ValueError("目标标签不能为空字符串")

    correct_sequences = 0
    total_edit_distance = 0
    total_target_characters = 0
    length_totals: dict[int, int] = {}
    length_correct: dict[int, int] = {}

    for expected, predicted in zip(
        expected_labels,
        predicted_labels,
        strict=True,
    ):
        is_correct = expected == predicted
        length = len(expected)
        correct_sequences += int(is_correct)
        total_edit_distance += levenshtein_distance(expected, predicted)
        total_target_characters += length
        length_totals[length] = length_totals.get(length, 0) + 1
        length_correct[length] = length_correct.get(length, 0) + int(is_correct)

    per_length = {
        length: LengthEvaluation(
            length=length,
            total=total,
            correct=length_correct[length],
            accuracy=length_correct[length] / total,
        )
        for length, total in sorted(length_totals.items())
    }
    total_sequences = len(expected_labels)
    return SequenceEvaluation(
        total_sequences=total_sequences,
        correct_sequences=correct_sequences,
        exact_match_accuracy=correct_sequences / total_sequences,
        total_edit_distance=total_edit_distance,
        total_target_characters=total_target_characters,
        character_error_rate=total_edit_distance / total_target_characters,
        per_length=per_length,
    )
