from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AccuracyCounts:
    """保存一个批次中的整图与单字符预测统计。"""

    correct_captchas: int
    correct_characters: int
    total_captchas: int
    total_characters: int

    @property
    def exact_match_accuracy(self) -> float:
        """返回整张验证码全部预测正确的比例。"""
        if self.total_captchas == 0:
            return 0.0
        return self.correct_captchas / self.total_captchas

    @property
    def character_accuracy(self) -> float:
        """返回所有字符位置中预测正确的比例。"""
        if self.total_characters == 0:
            return 0.0
        return self.correct_characters / self.total_characters


def calculate_accuracy_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_char: int,
    num_class: int,
) -> AccuracyCounts:
    """统计一个批次的整图正确数和单字符正确数。

    logits 可以是 [B, num_char * num_class] 或 [B, num_char, num_class]；
    target 使用相同两种形状之一，并以 one-hot 形式保存真实类别。
    """
    predicted = logits.reshape(-1, num_char, num_class).argmax(dim=2)
    expected = target.reshape(-1, num_char, num_class).argmax(dim=2)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"预测和标签形状不一致：{tuple(predicted.shape)} != {tuple(expected.shape)}"
        )

    matches = predicted == expected
    return AccuracyCounts(
        correct_captchas=matches.all(dim=1).sum().item(),
        correct_characters=matches.sum().item(),
        total_captchas=matches.size(0),
        total_characters=matches.numel(),
    )
