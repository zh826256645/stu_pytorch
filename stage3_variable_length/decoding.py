"""CTC 贪心路径与 prefix beam search 解码工具。"""

import math
from collections.abc import Sequence

import torch


def decode_ctc_path(
    path: Sequence[int],
    characters: str,
    blank_index: int,
) -> str:
    """合并连续重复类别并删除 blank，得到一条 CTC 路径的文本。"""
    decoded_characters: list[str] = []
    previous_index: int | None = None

    for index in path:
        if index != previous_index and index != blank_index:
            character_index = index if index < blank_index else index - 1
            decoded_characters.append(characters[character_index])
        previous_index = index

    return "".join(decoded_characters)


def _log_add(*values: float) -> float:
    finite_values = [value for value in values if value != float("-inf")]
    if not finite_values:
        return float("-inf")
    maximum = max(finite_values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite_values))


def _prefix_to_text(
    prefix: tuple[int, ...],
    characters: str,
    blank_index: int,
) -> str:
    return "".join(
        characters[index if index < blank_index else index - 1] for index in prefix
    )


def decode_ctc_prefix_beam_search(
    logits: torch.Tensor,
    characters: str,
    blank_index: int,
    *,
    beam_width: int = 10,
) -> str:
    """在 log-space 中汇总同一前缀的 CTC 路径并返回最高概率文本。"""
    if logits.ndim != 2:
        raise ValueError("单样本 logits 必须是 [T, C]")
    if beam_width <= 0:
        raise ValueError("beam_width 必须大于 0")
    class_count = logits.size(1)
    if class_count != len(characters) + 1:
        raise ValueError("logits 类别数必须等于字符数加一个 blank")
    if not 0 <= blank_index < class_count:
        raise ValueError("blank_index 超出类别范围")

    log_probabilities = logits.log_softmax(dim=1).detach().cpu().tolist()
    negative_infinity = float("-inf")
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, negative_infinity)}

    for time_step_probabilities in log_probabilities:
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (blank_probability, nonblank_probability) in beams.items():
            for class_index, class_probability in enumerate(time_step_probabilities):
                if class_index == blank_index:
                    current_blank, current_nonblank = next_beams.get(
                        prefix,
                        (negative_infinity, negative_infinity),
                    )
                    next_beams[prefix] = (
                        _log_add(
                            current_blank,
                            blank_probability + class_probability,
                            nonblank_probability + class_probability,
                        ),
                        current_nonblank,
                    )
                    continue

                if prefix and class_index == prefix[-1]:
                    current_blank, current_nonblank = next_beams.get(
                        prefix,
                        (negative_infinity, negative_infinity),
                    )
                    next_beams[prefix] = (
                        current_blank,
                        _log_add(
                            current_nonblank,
                            nonblank_probability + class_probability,
                        ),
                    )
                    extended_prefix = prefix + (class_index,)
                    extended_blank, extended_nonblank = next_beams.get(
                        extended_prefix,
                        (negative_infinity, negative_infinity),
                    )
                    next_beams[extended_prefix] = (
                        extended_blank,
                        _log_add(
                            extended_nonblank,
                            blank_probability + class_probability,
                        ),
                    )
                    continue

                extended_prefix = prefix + (class_index,)
                extended_blank, extended_nonblank = next_beams.get(
                    extended_prefix,
                    (negative_infinity, negative_infinity),
                )
                next_beams[extended_prefix] = (
                    extended_blank,
                    _log_add(
                        extended_nonblank,
                        blank_probability + class_probability,
                        nonblank_probability + class_probability,
                    ),
                )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _log_add(*item[1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(beams, key=lambda prefix: _log_add(*beams[prefix]))
    return _prefix_to_text(best_prefix, characters, blank_index)


def decode_ctc_prefix_beam_search_batch(
    logits: torch.Tensor,
    characters: str,
    blank_index: int,
    *,
    beam_width: int = 10,
) -> list[str]:
    """对 batch-first logits `[B, T, C]` 逐样本执行 prefix beam search。"""
    if logits.ndim != 3:
        raise ValueError("batch logits 必须是 [B, T, C]")
    return [
        decode_ctc_prefix_beam_search(
            sample_logits,
            characters,
            blank_index,
            beam_width=beam_width,
        )
        for sample_logits in logits
    ]
