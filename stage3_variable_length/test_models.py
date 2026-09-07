"""可变长度 CNN 输出接口的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_models
"""

import torch
from torch import nn

from .models import DEFAULT_CHARACTERS, VariableLengthCaptchaCNN

model = VariableLengthCaptchaCNN(characters=DEFAULT_CHARACTERS)
model.eval()

images = torch.randn(2, 3, 100, 180)
with torch.inference_mode():
    logits = model(images)

assert logits.shape[0] == 2
assert logits.shape[1] == 22
assert logits.shape[2] == len(DEFAULT_CHARACTERS) + 1
assert model.sequence_model == "cnn"

bilstm_model = VariableLengthCaptchaCNN(
    characters=DEFAULT_CHARACTERS,
    sequence_model="bilstm",
    lstm_hidden_size=64,
).eval()
with torch.inference_mode():
    bilstm_logits = bilstm_model(images)
assert bilstm_logits.shape == logits.shape
assert bilstm_model.sequence_model == "bilstm"
assert isinstance(bilstm_model.sequence_encoder, nn.LSTM)
assert bilstm_model.sequence_encoder.bidirectional
assert bilstm_model.sequence_encoder.hidden_size == 64
assert sum(parameter.numel() for parameter in bilstm_model.parameters()) == 175_573

wide_sequence_model = VariableLengthCaptchaCNN(
    characters=DEFAULT_CHARACTERS,
    sequence_model="bilstm",
    lstm_hidden_size=64,
    sequence_width=45,
).eval()
with torch.inference_mode():
    wide_sequence_logits = wide_sequence_model(images)
assert wide_sequence_logits.shape == (2, 45, len(DEFAULT_CHARACTERS) + 1)
assert wide_sequence_model.sequence_width == 45
assert (
    sum(parameter.numel() for parameter in wide_sequence_model.parameters()) == 175_573
)

for invalid_sequence_model in ("lstm", "transformer"):
    try:
        VariableLengthCaptchaCNN(sequence_model=invalid_sequence_model)
    except ValueError:
        pass
    else:
        raise AssertionError("不支持的序列模型应被拒绝")

for invalid_sequence_width in (21, 44, 46):
    try:
        VariableLengthCaptchaCNN(sequence_width=invalid_sequence_width)
    except ValueError:
        pass
    else:
        raise AssertionError("不支持的序列宽度应被拒绝")

print(
    f"CNN T={logits.shape[1]} | BiLSTM T={bilstm_logits.shape[1]} | "
    f"高分辨率BiLSTM T={wide_sequence_logits.shape[1]}"
)
