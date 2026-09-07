"""正式 CNN + CTC 训练循环的最小冒烟测试。

运行方式：
    uv run python -m stage3_variable_length.test_train
"""

import json
import math
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from .data import TinyCaptchaDataset
from .models import DEFAULT_CHARACTERS, VariableLengthCaptchaCNN
from .train import (
    BaselineTrainingConfig,
    build_length_sample_weights,
    build_lr_scheduler,
    evaluate_model,
    evaluate_model_with_batch_statistics,
    parse_args,
    recalibrate_batchnorm_model,
    run_training_epoch,
    run_training_epoch_with_diagnostics,
    train_baseline,
)

labels = ["aa", "a7b", "b8x2p6"]
data_loader = DataLoader(
    TinyCaptchaDataset(labels, seed=0),
    batch_size=3,
    shuffle=False,
)
torch.manual_seed(0)
model = VariableLengthCaptchaCNN()
diagnostic_model = VariableLengthCaptchaCNN()
diagnostic_model.load_state_dict(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
diagnostic_optimizer = torch.optim.Adam(diagnostic_model.parameters(), lr=0.001)
device = torch.device("cpu")

training_loss = run_training_epoch(model, data_loader, optimizer, device)
assert math.isfinite(training_loss)
assert training_loss > 0

diagnostics = run_training_epoch_with_diagnostics(
    diagnostic_model,
    data_loader,
    diagnostic_optimizer,
    device,
)
assert diagnostics.batch_count == 1
assert diagnostics.max_batch_loss_index == 1
assert diagnostics.max_gradient_norm_index == 1
assert math.isfinite(diagnostics.average_loss)
assert math.isfinite(diagnostics.max_batch_loss)
assert math.isfinite(diagnostics.max_gradient_norm)
assert diagnostics.last_batch_loss == diagnostics.max_batch_loss
assert diagnostics.last_gradient_norm == diagnostics.max_gradient_norm
assert diagnostics.nonfinite_loss_count == 0
assert diagnostics.nonfinite_gradient_count == 0
assert diagnostics.average_loss == training_loss
for parameter, diagnostic_parameter in zip(
    model.parameters(),
    diagnostic_model.parameters(),
    strict=True,
):
    assert torch.equal(parameter, diagnostic_parameter)
assert parse_args(["--diagnose-batches"]).diagnose_batches is True

validation_loss, result, comparisons = evaluate_model(model, data_loader, device)
assert math.isfinite(validation_loss)
assert result.total_sequences == 3
assert result.total_target_characters == 11
assert set(result.per_length) == {2, 3, 6}
assert [expected for expected, _predicted in comparisons] == labels

model_state_before_batch_statistics = {
    name: value.detach().clone() for name, value in model.state_dict().items()
}
model_mode_before_batch_statistics = model.training
batch_stat_loss, batch_stat_result, _batch_stat_comparisons = (
    evaluate_model_with_batch_statistics(model, data_loader, device)
)
assert math.isfinite(batch_stat_loss)
assert batch_stat_result.total_sequences == 3
assert model.training == model_mode_before_batch_statistics
for name, value in model.state_dict().items():
    assert torch.equal(value, model_state_before_batch_statistics[name])
assert parse_args(["--diagnose-batchnorm"]).diagnose_batchnorm is True

calibrated_model = recalibrate_batchnorm_model(model, data_loader, device)
assert not calibrated_model.training
for name, value in model.state_dict().items():
    assert torch.equal(value, model_state_before_batch_statistics[name])
assert any(
    not torch.equal(value, model_state_before_batch_statistics[name])
    for name, value in calibrated_model.state_dict().items()
    if "running_mean" in name or "running_var" in name
)
assert parse_args(["--recalibrate-batchnorm"]).recalibrate_batchnorm is True
assert parse_args(["--weight-decay", "0.0001"]).weight_decay == 0.0001
assert parse_args(["--reduce-lr-on-plateau"]).reduce_lr_on_plateau is True
assert build_length_sample_weights(
    ["ab", "abcdef", "abc", "654321"],
    length_6_weight=2.0,
).tolist() == [1.0, 2.0, 1.0, 2.0]
weighted_args = parse_args(["--length-6-weight", "2.0"])
assert weighted_args.length_6_weight == 2.0
loss_weighted_args = parse_args(["--length-6-loss-weight", "2.0"])
assert loss_weighted_args.length_6_loss_weight == 2.0
bilstm_args = parse_args(
    [
        "--sequence-model",
        "bilstm",
        "--lstm-hidden-size",
        "64",
        "--sequence-width",
        "45",
    ]
)
assert bilstm_args.sequence_model == "bilstm"
assert bilstm_args.lstm_hidden_size == 64
assert bilstm_args.sequence_width == 45
scheduler_model = torch.nn.Linear(1, 1)
scheduler_optimizer = torch.optim.Adam(scheduler_model.parameters(), lr=0.001)
scheduler = build_lr_scheduler(scheduler_optimizer)
for scheduler_validation_loss in (1.0, 1.1, 1.1):
    scheduler.step(scheduler_validation_loss)
assert abs(scheduler_optimizer.param_groups[0]["lr"] - 0.0003) < 1e-12
try:
    train_baseline(
        BaselineTrainingConfig(
            evaluate_train=True,
            diagnose_batchnorm=True,
            recalibrate_batchnorm=True,
        ),
        train_dir=Path("unused-train"),
        validation_dir=Path("unused-validation"),
    )
except ValueError as error:
    assert "不能同时启用" in str(error)
else:
    raise AssertionError("BatchNorm诊断与校准模式应互斥")

try:
    train_baseline(
        BaselineTrainingConfig(weight_decay=-0.0001),
        train_dir=Path("unused-train"),
        validation_dir=Path("unused-validation"),
    )
except ValueError as error:
    assert "weight_decay 必须大于等于 0" in str(error)
else:
    raise AssertionError("负权重衰减应被拒绝")


def write_manifest_split(
    root: Path,
    split: str,
    split_labels: list[str],
    seed: int,
) -> Path:
    split_directory = root / split
    split_directory.mkdir()
    dataset = TinyCaptchaDataset(split_labels, seed=seed)
    records = []
    for index, label in enumerate(split_labels):
        file_name = f"sample_{index}.png"
        to_pil_image(dataset[index][0]).save(split_directory / file_name)
        records.append({"file": file_name, "label": label, "length": len(label)})

    manifest = {
        "version": 1,
        "split": split,
        "count": len(records),
        "width": 180,
        "height": 100,
        "characters": DEFAULT_CHARACTERS,
        "files": records,
    }
    (split_directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return split_directory


with TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    train_dir = write_manifest_split(root, "train", labels, seed=1)
    validation_dir = write_manifest_split(
        root,
        "validation",
        ["27", "33m", "xxy23"],
        seed=2,
    )
    model_path = root / "checkpoints" / "model.pth"
    output = StringIO()
    with redirect_stdout(output):
        train_baseline(
            BaselineTrainingConfig(
                epochs=1,
                batch_size=3,
                learning_rate=0.001,
                seed=0,
                device="cpu",
                evaluate_train=True,
                diagnose_batches=True,
                diagnose_batchnorm=True,
            ),
            train_dir=train_dir,
            validation_dir=validation_dir,
            model_path=model_path,
        )
    logs = output.getvalue()
    assert "训练评价" in logs
    assert "泛化差距" in logs
    assert "批次诊断" in logs
    assert "max_grad_norm" in logs
    assert "BatchNorm差分" in logs
    assert "batch-stat验证按长度" in logs
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    assert checkpoint["version"] == 1
    assert checkpoint["epoch"] == 1
    assert checkpoint["characters"] == DEFAULT_CHARACTERS
    assert "model_state_dict" in checkpoint

    calibrated_model_path = root / "calibrated_checkpoints" / "model.pth"
    calibrated_output = StringIO()
    with redirect_stdout(calibrated_output):
        train_baseline(
            BaselineTrainingConfig(
                epochs=1,
                batch_size=3,
                learning_rate=0.001,
                seed=0,
                device="cpu",
                evaluate_train=True,
                recalibrate_batchnorm=True,
                weight_decay=0.0001,
                reduce_lr_on_plateau=True,
            ),
            train_dir=train_dir,
            validation_dir=validation_dir,
            model_path=calibrated_model_path,
        )
    calibrated_logs = calibrated_output.getvalue()
    assert "BatchNorm校准" in calibrated_logs
    assert "校准前验证" in calibrated_logs
    assert "权重衰减：0.0001" in calibrated_logs
    assert "自动降低学习率：True" in calibrated_logs
    assert "学习率=0.001000" in calibrated_logs
    calibrated_checkpoint = torch.load(
        calibrated_model_path,
        map_location="cpu",
        weights_only=True,
    )
    assert calibrated_checkpoint["training_config"]["recalibrate_batchnorm"] is True
    assert calibrated_checkpoint["training_config"]["weight_decay"] == 0.0001
    assert calibrated_checkpoint["training_config"]["reduce_lr_on_plateau"] is True
    assert not torch.equal(
        checkpoint["model_state_dict"]["sequence_classifier.weight"],
        calibrated_checkpoint["model_state_dict"]["sequence_classifier.weight"],
    )

    bilstm_model_path = root / "bilstm_checkpoints" / "model.pth"
    train_baseline(
        BaselineTrainingConfig(
            epochs=1,
            batch_size=3,
            learning_rate=0.001,
            seed=0,
            device="cpu",
            sequence_model="bilstm",
            lstm_hidden_size=64,
            sequence_width=45,
            length_6_loss_weight=2.0,
            evaluate_train=True,
            recalibrate_batchnorm=True,
            reduce_lr_on_plateau=True,
        ),
        train_dir=train_dir,
        validation_dir=validation_dir,
        model_path=bilstm_model_path,
    )
    bilstm_checkpoint = torch.load(
        bilstm_model_path,
        map_location="cpu",
        weights_only=True,
    )
    assert bilstm_checkpoint["model_config"] == {
        "sequence_model": "bilstm",
        "lstm_hidden_size": 64,
        "sequence_width": 45,
    }
    assert bilstm_checkpoint["training_config"]["length_6_weight"] == 1.0
    assert bilstm_checkpoint["training_config"]["length_6_loss_weight"] == 2.0

print(
    f"正式训练与 checkpoint 冒烟测试通过：train_loss={training_loss:.4f}, "
    f"validation_loss={validation_loss:.4f}"
)
