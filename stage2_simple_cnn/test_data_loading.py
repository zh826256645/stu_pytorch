"""真实/合成固定比例批次采样器检查。

运行方式：
    uv run python -m stage2_simple_cnn.test_data_loading
"""

import math

import torch

from .data_loading import MixedBatchSampler


def collect_batches(seed: int, ratio: float) -> list[list[int]]:
    sampler = MixedBatchSampler(
        real_size=800,
        synthetic_size=4000,
        batch_size=16,
        synthetic_ratio=ratio,
        generator=torch.Generator().manual_seed(seed),
    )
    return list(sampler)


quarter_sampler = MixedBatchSampler(
    real_size=800,
    synthetic_size=4000,
    batch_size=16,
    synthetic_ratio=0.25,
    generator=torch.Generator().manual_seed(7),
)
assert quarter_sampler.real_per_batch == 12
assert quarter_sampler.synthetic_per_batch == 4
assert quarter_sampler.actual_synthetic_ratio == 0.25
assert len(quarter_sampler) == math.ceil(800 / 12)

quarter_batches = list(quarter_sampler)
assert all(len(batch) == 16 for batch in quarter_batches)
assert all(sum(index < 800 for index in batch) == 12 for batch in quarter_batches)
assert all(sum(index >= 800 for index in batch) == 4 for batch in quarter_batches)
assert set(range(800)).issubset(
    {index for batch in quarter_batches for index in batch if index < 800}
)

half_batches = collect_batches(seed=11, ratio=0.5)
assert all(sum(index < 800 for index in batch) == 8 for batch in half_batches)
assert all(sum(index >= 800 for index in batch) == 8 for batch in half_batches)
assert half_batches == collect_batches(seed=11, ratio=0.5)
assert half_batches != collect_batches(seed=12, ratio=0.5)

rounded_sampler = MixedBatchSampler(
    real_size=5,
    synthetic_size=7,
    batch_size=3,
    synthetic_ratio=0.5,
)
assert rounded_sampler.synthetic_per_batch == 2
assert rounded_sampler.real_per_batch == 1

invalid_arguments = (
    {"real_size": 0, "synthetic_size": 1, "batch_size": 2, "synthetic_ratio": 0.5},
    {"real_size": 1, "synthetic_size": 0, "batch_size": 2, "synthetic_ratio": 0.5},
    {"real_size": 1, "synthetic_size": 1, "batch_size": 1, "synthetic_ratio": 0.5},
    {"real_size": 1, "synthetic_size": 1, "batch_size": 2, "synthetic_ratio": 0},
    {"real_size": 1, "synthetic_size": 1, "batch_size": 2, "synthetic_ratio": 1},
)
for arguments in invalid_arguments:
    try:
        MixedBatchSampler(**arguments)
    except ValueError:
        pass
    else:
        raise AssertionError(f"无效参数应抛出 ValueError：{arguments}")

print("混合批次采样器测试通过")
