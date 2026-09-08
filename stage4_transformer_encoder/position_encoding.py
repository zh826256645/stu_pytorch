"""第二步：固定正弦/余弦位置编码。

运行检查：uv run python -m stage4_transformer_encoder.position_encoding
"""

import math

import torch


def add_position_encoding(sequence: torch.Tensor) -> torch.Tensor:
    """为浮点序列 [B, T, D] 加入位置编码；本教学版本要求 D 为正偶数。"""
    if sequence.ndim != 3 or not sequence.is_floating_point():
        raise ValueError("sequence 必须是 [B, T, D] 浮点张量")
    _, length, dimension = sequence.shape
    if dimension == 0 or dimension % 2:
        raise ValueError("特征维度 D 必须是正偶数")

    # 每行是一个横向位置，每列是一种频率：[T, 1] × [D/2]。
    positions = torch.arange(length, device=sequence.device, dtype=torch.float32)
    channels = torch.arange(0, dimension, 2, device=sequence.device).float()
    angles = positions[:, None] / (10000.0 ** (channels / dimension))
    encoding = torch.empty(length, dimension, device=sequence.device)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles.cos()
    # [T, D] 在 batch 维广播：不同图片使用同一套位置编码。
    return sequence + encoding.to(dtype=sequence.dtype)


def demo() -> None:
    sequence = torch.zeros(2, 22, 64, requires_grad=True)
    encoded = add_position_encoding(sequence)
    assert encoded.shape == sequence.shape
    torch.testing.assert_close(encoded[0], encoded[1])
    torch.testing.assert_close(encoded[0, 0, 0::2], torch.zeros(32))
    torch.testing.assert_close(encoded[0, 0, 1::2], torch.ones(32))
    torch.testing.assert_close(
        encoded[0, 1, :2], torch.tensor([math.sin(1.0), math.cos(1.0)])
    )
    assert not torch.equal(encoded[0, 0], encoded[0, 1])
    encoded.sum().backward()
    torch.testing.assert_close(sequence.grad, torch.ones_like(sequence))
    torch.testing.assert_close(sequence, torch.zeros_like(sequence))
    for invalid in (
        torch.zeros(22, 64),
        torch.zeros(1, 22, 3),
        torch.zeros(1, 22, 0),
        torch.zeros(1, 22, 64, dtype=torch.int64),
    ):
        try:
            add_position_encoding(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("应拒绝不符合输入约定的张量")
    print("位置编码检查通过：形状、位置差异、批次共享、梯度与输入校验。")
    print("位置 0 的前 4 维：", encoded[0, 0, :4].detach().tolist())
    print("位置 1 的前 4 维：", encoded[0, 1, :4].detach().tolist())


if __name__ == "__main__":
    demo()
