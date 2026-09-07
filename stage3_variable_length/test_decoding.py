"""CTC 贪心路径与 prefix beam search 的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_decoding
"""

import torch

from .decoding import (
    decode_ctc_path,
    decode_ctc_prefix_beam_search,
    decode_ctc_prefix_beam_search_batch,
)

characters = "ab7"
blank_index = len(characters)

# 连续的相同类别来自相邻时间步，应合并为一个字符。
path = [blank_index, 0, 0, blank_index, 1, 1, blank_index, 2]
assert decode_ctc_path(path, characters, blank_index) == "ab7"

# 没有 blank 分隔的连续相同类别会被折叠成一个字符。
assert decode_ctc_path([0, 0, 0], characters, blank_index) == "a"

# blank 会打断连续重复，因此同一个字符可以在结果中连续出现两次。
assert (
    decode_ctc_path(
        [0, 0, blank_index, 0, 0],
        characters,
        blank_index,
    )
    == "aa"
)

# 模型在所有时间步都预测 blank 时，解码结果为空字符串。
assert (
    decode_ctc_path(
        [blank_index, blank_index, blank_index],
        characters,
        blank_index,
    )
    == ""
)

beam_characters = "ab"
beam_blank_index = len(beam_characters)
probabilities = torch.tensor(
    [
        [0.101, 0.467, 0.432],
        [0.872, 0.120, 0.008],
        [0.312, 0.158, 0.530],
        [0.921, 0.075, 0.004],
    ],
    dtype=torch.float64,
)
beam_logits = probabilities.log()
greedy_path = beam_logits.argmax(dim=1).tolist()
assert decode_ctc_path(greedy_path, beam_characters, beam_blank_index) == "baa"
assert (
    decode_ctc_prefix_beam_search(
        beam_logits,
        beam_characters,
        beam_blank_index,
        beam_width=10,
    )
    == "aa"
)
assert decode_ctc_prefix_beam_search_batch(
    torch.stack([beam_logits, beam_logits]),
    beam_characters,
    beam_blank_index,
    beam_width=10,
) == ["aa", "aa"]

try:
    decode_ctc_prefix_beam_search(
        beam_logits,
        beam_characters,
        beam_blank_index,
        beam_width=0,
    )
except ValueError:
    pass
else:
    raise AssertionError("beam_width=0 应被拒绝")

print("CTC贪心路径与prefix beam search测试通过")
