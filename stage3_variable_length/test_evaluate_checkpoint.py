"""最佳 checkpoint 只读评估入口的最小行为测试。

运行方式：
    uv run python -m stage3_variable_length.test_evaluate_checkpoint
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .evaluate_checkpoint import load_checkpoint_model, parse_args
from .models import DEFAULT_CHARACTERS, VariableLengthCaptchaCNN

beam_args = parse_args(["--decoder", "beam", "--beam-width", "10"])
assert beam_args.decoder == "beam"
assert beam_args.beam_width == 10

torch.manual_seed(0)
source_model = VariableLengthCaptchaCNN().eval()
images = torch.randn(2, 3, 100, 180)
with torch.inference_mode():
    expected_logits = source_model(images)

with TemporaryDirectory() as temporary_directory:
    checkpoint_path = Path(temporary_directory) / "model.pth"
    torch.save(
        {
            "version": 1,
            "model_state_dict": source_model.state_dict(),
            "characters": DEFAULT_CHARACTERS,
            "blank_index": len(DEFAULT_CHARACTERS),
            "epoch": 3,
        },
        checkpoint_path,
    )
    loaded_model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        torch.device("cpu"),
    )
    with torch.inference_mode():
        actual_logits = loaded_model(images)

    assert checkpoint["epoch"] == 3
    assert loaded_model.sequence_model == "cnn"
    assert not loaded_model.training
    assert torch.equal(actual_logits, expected_logits)

    bilstm_source_model = VariableLengthCaptchaCNN(
        sequence_model="bilstm",
        lstm_hidden_size=64,
        sequence_width=45,
    ).eval()
    with torch.inference_mode():
        expected_bilstm_logits = bilstm_source_model(images)
    bilstm_checkpoint_path = Path(temporary_directory) / "bilstm_model.pth"
    torch.save(
        {
            "version": 1,
            "model_state_dict": bilstm_source_model.state_dict(),
            "characters": DEFAULT_CHARACTERS,
            "blank_index": len(DEFAULT_CHARACTERS),
            "model_config": {
                "sequence_model": "bilstm",
                "lstm_hidden_size": 64,
                "sequence_width": 45,
            },
            "epoch": 4,
        },
        bilstm_checkpoint_path,
    )
    loaded_bilstm_model, bilstm_checkpoint = load_checkpoint_model(
        bilstm_checkpoint_path,
        torch.device("cpu"),
    )
    with torch.inference_mode():
        actual_bilstm_logits = loaded_bilstm_model(images)
    assert bilstm_checkpoint["epoch"] == 4
    assert loaded_bilstm_model.sequence_model == "bilstm"
    assert loaded_bilstm_model.lstm_hidden_size == 64
    assert loaded_bilstm_model.sequence_width == 45
    assert actual_bilstm_logits.shape[1] == 45
    assert torch.equal(actual_bilstm_logits, expected_bilstm_logits)

print("CNN与BiLSTM checkpoint只读加载测试通过")
