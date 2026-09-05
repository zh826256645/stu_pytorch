"""第二阶段真实数据与合成数据的固定比例批次采样。"""

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class MixedBatchSampler(Sampler[list[int]]):
    """每个批次按固定数量混合真实样本和合成样本。

    真实数据位于组合数据集的前半部分，索引范围为
    ``[0, real_size)``；合成数据紧随其后。每轮至少完整遍历一次
    真实训练集，较小的数据源在需要时重新洗牌并循环采样。
    """

    def __init__(
        self,
        *,
        real_size: int,
        synthetic_size: int,
        batch_size: int,
        synthetic_ratio: float,
        generator: torch.Generator | None = None,
    ) -> None:
        if real_size <= 0:
            raise ValueError("real_size 必须大于 0")
        if synthetic_size <= 0:
            raise ValueError("synthetic_size 必须大于 0")
        if batch_size < 2:
            raise ValueError("混合批次的 batch_size 必须至少为 2")
        if not 0 < synthetic_ratio < 1:
            raise ValueError("synthetic_ratio 必须大于 0 且小于 1")

        synthetic_per_batch = round(batch_size * synthetic_ratio)
        synthetic_per_batch = max(1, min(batch_size - 1, synthetic_per_batch))

        self.real_size = real_size
        self.synthetic_size = synthetic_size
        self.batch_size = batch_size
        self.synthetic_per_batch = synthetic_per_batch
        self.real_per_batch = batch_size - synthetic_per_batch
        self.generator = generator
        self.num_batches = math.ceil(real_size / self.real_per_batch)

    @property
    def actual_synthetic_ratio(self) -> float:
        """返回取整后的实际批次合成数据比例。"""
        return self.synthetic_per_batch / self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def _draw_indices(self, *, size: int, count: int, offset: int) -> list[int]:
        indices = []
        while len(indices) < count:
            permutation = torch.randperm(size, generator=self.generator).tolist()
            indices.extend(index + offset for index in permutation)
        return indices[:count]

    def __iter__(self) -> Iterator[list[int]]:
        total_real = self.num_batches * self.real_per_batch
        total_synthetic = self.num_batches * self.synthetic_per_batch
        real_indices = self._draw_indices(
            size=self.real_size,
            count=total_real,
            offset=0,
        )
        synthetic_indices = self._draw_indices(
            size=self.synthetic_size,
            count=total_synthetic,
            offset=self.real_size,
        )

        for batch_index in range(self.num_batches):
            real_start = batch_index * self.real_per_batch
            synthetic_start = batch_index * self.synthetic_per_batch
            batch = [
                *real_indices[real_start : real_start + self.real_per_batch],
                *synthetic_indices[
                    synthetic_start : synthetic_start + self.synthetic_per_batch
                ],
            ]
            order = torch.randperm(
                self.batch_size,
                generator=self.generator,
            ).tolist()
            yield [batch[index] for index in order]
