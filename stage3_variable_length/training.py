"""第三阶段 CTC 训练所需的最小工具。"""

from collections.abc import Sequence

import torch


def encode_ctc_targets(
    labels: Sequence[str],
    characters: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把不同长度的文本编码成 CTC 支持的一维拼接目标。"""
    character_to_index = {
        character: index for index, character in enumerate(characters)
    }
    encoded_targets = [
        character_to_index[character] for label in labels for character in label
    ]
    target_lengths = [len(label) for label in labels]

    return (
        torch.tensor(encoded_targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def compute_ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_index: int,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算按目标长度归一化、可选逐样本加权的 CTC 损失。"""
    batch_size, time_steps, _ = logits.shape

    # PyTorch CTCLoss 要求输入为 [T, B, C]，并且输入必须是 log 概率。
    log_probabilities = logits.log_softmax(dim=2).transpose(0, 1)
    input_lengths = torch.full(
        (batch_size,),
        time_steps,
        dtype=torch.long,
    )

    per_sample_losses = torch.nn.functional.ctc_loss(
        log_probabilities,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_index,
        reduction="none",
    )
    normalized_losses = per_sample_losses / target_lengths.to(
        device=per_sample_losses.device,
        dtype=per_sample_losses.dtype,
    )
    if sample_weights is None:
        return normalized_losses.mean()
    if sample_weights.ndim != 1 or sample_weights.numel() != batch_size:
        raise ValueError("sample_weights 必须为每个 batch 样本提供一个权重")
    weights = sample_weights.to(
        device=per_sample_losses.device,
        dtype=per_sample_losses.dtype,
    )
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weights 必须是有限正数")
    return (normalized_losses * weights).sum() / weights.sum()
