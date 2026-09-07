"""可变长度识别评价指标的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_metrics
"""

import math

from .metrics import evaluate_predictions, levenshtein_distance

assert levenshtein_distance("a7", "a7") == 0
assert levenshtein_distance("b8x2", "b8x") == 1
assert levenshtein_distance("b8x", "b8x2") == 1
assert levenshtein_distance("33m", "3am") == 1

result = evaluate_predictions(
    ["a7", "b8x2", "33m"],
    ["a7", "b8x", "3am"],
)
assert result.total_sequences == 3
assert result.correct_sequences == 1
assert math.isclose(result.exact_match_accuracy, 1 / 3)
assert result.total_edit_distance == 2
assert result.total_target_characters == 9
assert math.isclose(result.character_error_rate, 2 / 9)
assert result.per_length[2].total == 1
assert result.per_length[2].correct == 1
assert result.per_length[2].accuracy == 1.0
assert result.per_length[3].accuracy == 0.0
assert result.per_length[4].accuracy == 0.0

print("序列评价指标测试通过")
