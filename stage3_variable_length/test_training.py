"""可变长度 CTC 训练数据与损失的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_training
"""

import torch

from .models import DEFAULT_CHARACTERS
from .training import compute_ctc_loss, encode_ctc_targets

labels = ["a7", "b8x2"]
targets, target_lengths = encode_ctc_targets(labels, DEFAULT_CHARACTERS)

assert targets.dtype == torch.long
assert targets.tolist() == [7, 5, 8, 6, 18, 0]
assert target_lengths.dtype == torch.long
assert target_lengths.tolist() == [2, 4]

torch.manual_seed(0)
logits = torch.randn(
    len(labels),
    22,
    len(DEFAULT_CHARACTERS) + 1,
    requires_grad=True,
)
loss = compute_ctc_loss(
    logits,
    targets,
    target_lengths,
    blank_index=len(DEFAULT_CHARACTERS),
)
assert loss.ndim == 0
assert torch.isfinite(loss)
assert loss.item() > 0

unit_weights_loss = compute_ctc_loss(
    logits,
    targets,
    target_lengths,
    blank_index=len(DEFAULT_CHARACTERS),
    sample_weights=torch.ones(len(labels)),
)
assert torch.allclose(unit_weights_loss, loss)

first_loss = compute_ctc_loss(
    logits[0:1],
    targets[:2],
    target_lengths[:1],
    blank_index=len(DEFAULT_CHARACTERS),
)
second_loss = compute_ctc_loss(
    logits[1:2],
    targets[2:],
    target_lengths[1:],
    blank_index=len(DEFAULT_CHARACTERS),
)
weighted_loss = compute_ctc_loss(
    logits,
    targets,
    target_lengths,
    blank_index=len(DEFAULT_CHARACTERS),
    sample_weights=torch.tensor([1.0, 2.0]),
)
expected_weighted_loss = (first_loss + 2 * second_loss) / 3
assert torch.allclose(weighted_loss, expected_weighted_loss, atol=1e-6)

loss.backward()
assert logits.grad is not None
assert logits.grad.shape == logits.shape
assert torch.isfinite(logits.grad).all()

print(f"CTC 训练数据与损失测试通过：loss={loss.item():.4f}")
