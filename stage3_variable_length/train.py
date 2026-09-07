"""第三阶段正式 CNN + CTC 基线训练入口。

运行方式：
    uv run python -m stage3_variable_length.train
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import ManifestCaptchaDataset
from .decoding import decode_ctc_path, decode_ctc_prefix_beam_search_batch
from .metrics import SequenceEvaluation, evaluate_predictions
from .models import (
    DEFAULT_CHARACTERS,
    DEFAULT_LSTM_HIDDEN_SIZE,
    DEFAULT_SEQUENCE_WIDTH,
    SUPPORTED_SEQUENCE_MODELS,
    SUPPORTED_SEQUENCE_WIDTHS,
    VariableLengthCaptchaCNN,
)
from .training import compute_ctc_loss, encode_ctc_targets

DEFAULT_TRAIN_DIR = Path("data/stage3_variable_length/train")
DEFAULT_VALIDATION_DIR = Path("data/stage3_variable_length/validation")
DEFAULT_MODEL_PATH = Path("stage3_variable_length/checkpoints/model.pth")
PLATEAU_FACTOR = 0.3
PLATEAU_PATIENCE = 1
MIN_LEARNING_RATE = 1e-5
SUPPORTED_DECODERS = ("greedy", "beam")
DEFAULT_BEAM_WIDTH = 10


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """第一版正式 CNN + CTC 基线训练参数。"""

    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    sequence_model: str = "cnn"
    lstm_hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE
    sequence_width: int = DEFAULT_SEQUENCE_WIDTH
    length_6_weight: float = 1.0
    length_6_loss_weight: float = 1.0
    seed: int = 0
    device: str = "auto"
    evaluate_train: bool = False
    diagnose_batches: bool = False
    diagnose_batchnorm: bool = False
    recalibrate_batchnorm: bool = False
    reduce_lr_on_plateau: bool = False


@dataclass(frozen=True)
class TrainingEpochDiagnostics:
    """一轮训练中用于定位参数跳变的 batch 汇总。"""

    average_loss: float
    batch_count: int
    max_batch_loss: float
    max_batch_loss_index: int
    last_batch_loss: float
    max_gradient_norm: float
    max_gradient_norm_index: int
    last_gradient_norm: float
    nonfinite_loss_count: int
    nonfinite_gradient_count: int


def choose_device(requested: str) -> torch.device:
    """选择可用训练设备，并对显式请求进行检查。"""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前环境不可用 CUDA")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("当前环境不可用 MPS")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError(f"不支持的设备：{requested}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """验证 loss 连续不改善时自动降低学习率。"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )


def build_length_sample_weights(
    labels: Sequence[str],
    *,
    length_6_weight: float,
) -> torch.Tensor:
    """为六位标签分配可选采样权重，其余长度保持权重一。"""
    if length_6_weight <= 0:
        raise ValueError("length_6_weight 必须大于 0")
    return torch.tensor(
        [length_6_weight if len(label) == 6 else 1.0 for label in labels],
        dtype=torch.double,
    )


def decode_batch(
    logits: torch.Tensor,
    characters: str,
    blank_index: int,
    *,
    decoder: str = "greedy",
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> list[str]:
    """对 batch-first logits 执行指定的 CTC 解码。"""
    if decoder == "beam":
        return decode_ctc_prefix_beam_search_batch(
            logits,
            characters,
            blank_index,
            beam_width=beam_width,
        )
    if decoder != "greedy":
        supported = ", ".join(SUPPORTED_DECODERS)
        raise ValueError(f"decoder 必须是以下值之一：{supported}")
    paths = logits.argmax(dim=2).detach().cpu().tolist()
    return [decode_ctc_path(path, characters, blank_index) for path in paths]


def _gradient_l2_norm(model: VariableLengthCaptchaCNN) -> float:
    squared_norm = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum()
    return float(squared_norm.sqrt().item())


def _run_training_epoch(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    collect_diagnostics: bool,
    length_6_loss_weight: float,
) -> TrainingEpochDiagnostics:
    model.train()
    total_loss = 0.0
    total_loss_weight = 0.0
    batch_count = 0
    max_batch_loss = float("-inf")
    max_batch_loss_index = 0
    last_batch_loss = float("nan")
    max_gradient_norm = float("-inf")
    max_gradient_norm_index = 0
    last_gradient_norm = float("nan")
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        targets, target_lengths = encode_ctc_targets(labels, model.characters)
        sample_weights = (
            build_length_sample_weights(
                labels,
                length_6_weight=length_6_loss_weight,
            )
            if length_6_loss_weight != 1.0
            else None
        )
        logits = model(images.to(device))
        loss = compute_ctc_loss(
            logits,
            targets,
            target_lengths,
            blank_index=model.blank_index,
            sample_weights=sample_weights,
        )

        optimizer.zero_grad()
        loss.backward()
        gradient_norm = (
            _gradient_l2_norm(model) if collect_diagnostics else float("nan")
        )
        optimizer.step()

        batch_loss_weight = (
            float(sample_weights.sum().item())
            if sample_weights is not None
            else float(images.size(0))
        )
        batch_loss = float(loss.item())
        total_loss += batch_loss * batch_loss_weight
        total_loss_weight += batch_loss_weight
        batch_count = batch_index
        last_batch_loss = batch_loss
        if batch_loss > max_batch_loss:
            max_batch_loss = batch_loss
            max_batch_loss_index = batch_index
        if not math.isfinite(batch_loss):
            nonfinite_loss_count += 1

        if collect_diagnostics:
            last_gradient_norm = gradient_norm
            if gradient_norm > max_gradient_norm:
                max_gradient_norm = gradient_norm
                max_gradient_norm_index = batch_index
            if not math.isfinite(gradient_norm):
                nonfinite_gradient_count += 1

    if total_loss_weight == 0:
        raise ValueError("训练 DataLoader 不能为空")
    return TrainingEpochDiagnostics(
        average_loss=total_loss / total_loss_weight,
        batch_count=batch_count,
        max_batch_loss=max_batch_loss,
        max_batch_loss_index=max_batch_loss_index,
        last_batch_loss=last_batch_loss,
        max_gradient_norm=max_gradient_norm,
        max_gradient_norm_index=max_gradient_norm_index,
        last_gradient_norm=last_gradient_norm,
        nonfinite_loss_count=nonfinite_loss_count,
        nonfinite_gradient_count=nonfinite_gradient_count,
    )


def run_training_epoch(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    length_6_loss_weight: float = 1.0,
) -> float:
    """训练一轮并返回按样本平均的 CTC loss。"""
    return _run_training_epoch(
        model,
        data_loader,
        optimizer,
        device,
        collect_diagnostics=False,
        length_6_loss_weight=length_6_loss_weight,
    ).average_loss


def run_training_epoch_with_diagnostics(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    length_6_loss_weight: float = 1.0,
) -> TrainingEpochDiagnostics:
    """训练一轮并汇总 batch loss 与反向传播梯度范数。"""
    return _run_training_epoch(
        model,
        data_loader,
        optimizer,
        device,
        collect_diagnostics=True,
        length_6_loss_weight=length_6_loss_weight,
    )


def _evaluate_model_in_current_mode(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    device: torch.device,
    *,
    decoder: str = "greedy",
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> tuple[float, SequenceEvaluation, list[tuple[str, str]]]:
    total_loss = 0.0
    total_samples = 0
    expected_labels: list[str] = []
    predicted_labels: list[str] = []

    with torch.inference_mode():
        for images, labels in data_loader:
            targets, target_lengths = encode_ctc_targets(labels, model.characters)
            logits = model(images.to(device))
            loss = compute_ctc_loss(
                logits,
                targets,
                target_lengths,
                blank_index=model.blank_index,
            )
            predictions = decode_batch(
                logits,
                model.characters,
                model.blank_index,
                decoder=decoder,
                beam_width=beam_width,
            )

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            expected_labels.extend(labels)
            predicted_labels.extend(predictions)

    result = evaluate_predictions(expected_labels, predicted_labels)
    comparisons = list(zip(expected_labels, predicted_labels, strict=True))
    return total_loss / total_samples, result, comparisons


def evaluate_model(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    device: torch.device,
    *,
    decoder: str = "greedy",
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> tuple[float, SequenceEvaluation, list[tuple[str, str]]]:
    """使用冻结的 BatchNorm 运行统计计算模型评价指标。"""
    model.eval()
    return _evaluate_model_in_current_mode(
        model,
        data_loader,
        device,
        decoder=decoder,
        beam_width=beam_width,
    )


def _copy_model(
    model: VariableLengthCaptchaCNN,
    device: torch.device,
) -> VariableLengthCaptchaCNN:
    copied_model = VariableLengthCaptchaCNN(
        characters=model.characters,
        sequence_model=model.sequence_model,
        lstm_hidden_size=model.lstm_hidden_size,
        sequence_width=model.sequence_width,
    ).to(device)
    copied_model.load_state_dict(model.state_dict())
    return copied_model


def evaluate_model_with_batch_statistics(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, SequenceEvaluation, list[tuple[str, str]]]:
    """用模型副本和当前 batch 统计评价，且不修改原模型。"""
    diagnostic_model = _copy_model(model, device)
    diagnostic_model.eval()
    for module in diagnostic_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
    return _evaluate_model_in_current_mode(
        diagnostic_model,
        data_loader,
        device,
    )


def recalibrate_batchnorm_model(
    model: VariableLengthCaptchaCNN,
    calibration_loader: DataLoader,
    device: torch.device,
) -> VariableLengthCaptchaCNN:
    """复制模型，并只用训练数据重新构建 BatchNorm 运行统计。"""
    calibrated_model = _copy_model(model, device)
    calibrated_model.eval()
    batchnorm_modules = [
        module
        for module in calibrated_model.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
    for module in batchnorm_modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    calibration_batch_count = 0
    with torch.inference_mode():
        for images, _labels in calibration_loader:
            calibrated_model(images.to(device))
            calibration_batch_count += 1
    if calibration_batch_count == 0:
        raise ValueError("BatchNorm 校准 DataLoader 不能为空")

    calibrated_model.eval()
    return calibrated_model


def _is_better_validation_result(
    result: SequenceEvaluation,
    validation_loss: float,
    *,
    best_accuracy: float,
    best_character_error_rate: float,
    best_loss: float,
) -> bool:
    if result.exact_match_accuracy != best_accuracy:
        return result.exact_match_accuracy > best_accuracy
    if result.character_error_rate != best_character_error_rate:
        return result.character_error_rate < best_character_error_rate
    return validation_loss < best_loss


def _save_checkpoint(
    model: VariableLengthCaptchaCNN,
    model_path: Path,
    config: BaselineTrainingConfig,
    epoch: int,
    validation_loss: float,
    result: SequenceEvaluation,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: parameter.detach().cpu() for name, parameter in model.state_dict().items()
    }
    torch.save(
        {
            "version": 1,
            "model_state_dict": state_dict,
            "characters": model.characters,
            "blank_index": model.blank_index,
            "model_config": {
                "sequence_model": model.sequence_model,
                "lstm_hidden_size": model.lstm_hidden_size,
                "sequence_width": model.sequence_width,
            },
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_exact_match_accuracy": result.exact_match_accuracy,
            "validation_character_error_rate": result.character_error_rate,
            "validation_per_length_accuracy": {
                length: metrics.accuracy
                for length, metrics in result.per_length.items()
            },
            "training_config": asdict(config),
        },
        model_path,
    )


def _format_per_length(result: SequenceEvaluation) -> str:
    return " | ".join(
        f"{length}位={metrics.accuracy:.2%}"
        for length, metrics in result.per_length.items()
    )


def train_baseline(
    config: BaselineTrainingConfig,
    *,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """训练正式基线，并只保存验证表现最好的一个共享 checkpoint。"""
    if config.epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if config.weight_decay < 0:
        raise ValueError("weight_decay 必须大于等于 0")
    if config.sequence_model not in SUPPORTED_SEQUENCE_MODELS:
        supported = ", ".join(SUPPORTED_SEQUENCE_MODELS)
        raise ValueError(f"sequence_model 必须是以下值之一：{supported}")
    if config.lstm_hidden_size <= 0:
        raise ValueError("lstm_hidden_size 必须大于 0")
    if config.sequence_width not in SUPPORTED_SEQUENCE_WIDTHS:
        supported = ", ".join(str(value) for value in SUPPORTED_SEQUENCE_WIDTHS)
        raise ValueError(f"sequence_width 必须是以下值之一：{supported}")
    if config.length_6_weight <= 0:
        raise ValueError("length_6_weight 必须大于 0")
    if config.length_6_loss_weight <= 0:
        raise ValueError("length_6_loss_weight 必须大于 0")
    if config.diagnose_batchnorm and not config.evaluate_train:
        raise ValueError("diagnose_batchnorm 需要同时启用 evaluate_train")
    if config.recalibrate_batchnorm and not config.evaluate_train:
        raise ValueError("recalibrate_batchnorm 需要同时启用 evaluate_train")
    if config.diagnose_batchnorm and config.recalibrate_batchnorm:
        raise ValueError("diagnose_batchnorm 与 recalibrate_batchnorm 不能同时启用")

    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    train_dataset = ManifestCaptchaDataset(train_dir)
    validation_dataset = ManifestCaptchaDataset(validation_dir)
    if train_dataset.split != "train":
        raise ValueError("train_dir 的 manifest split 必须是 train")
    if validation_dataset.split != "validation":
        raise ValueError("validation_dir 的 manifest split 必须是 validation")
    if train_dataset.characters != validation_dataset.characters:
        raise ValueError("训练集和验证集字符表不一致")
    if train_dataset.characters != DEFAULT_CHARACTERS:
        raise ValueError("正式数据字符表与当前模型默认字符表不一致")

    sampling_generator = torch.Generator().manual_seed(config.seed)
    train_labels = [str(record["label"]) for record in train_dataset.records]
    training_sampler = (
        WeightedRandomSampler(
            build_length_sample_weights(
                train_labels,
                length_6_weight=config.length_6_weight,
            ),
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampling_generator,
        )
        if config.length_6_weight != 1.0
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=training_sampler is None,
        sampler=training_sampler,
        num_workers=0,
        generator=sampling_generator,
    )
    train_evaluation_loader = (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if config.evaluate_train
        else None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = VariableLengthCaptchaCNN(
        characters=train_dataset.characters,
        sequence_model=config.sequence_model,
        lstm_hidden_size=config.lstm_hidden_size,
        sequence_width=config.sequence_width,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer) if config.reduce_lr_on_plateau else None

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"使用设备：{device} | 参数量：{parameter_count:,} | "
        f"训练样本：{len(train_dataset)} | 验证样本：{len(validation_dataset)}"
    )
    print(
        f"最大轮数：{config.epochs} | batch size：{config.batch_size} | "
        f"学习率：{config.learning_rate} | 权重衰减：{config.weight_decay} | "
        f"seed：{config.seed}"
    )
    print(
        f"序列模型：{config.sequence_model} | "
        f"LSTM hidden size：{config.lstm_hidden_size} | "
        f"时间步：{config.sequence_width}"
    )
    print(
        f"六位采样权重：{config.length_6_weight} | "
        f"每轮抽样次数：{len(train_dataset)} | "
        f"有放回采样：{training_sampler is not None}"
    )
    print(f"六位CTC loss权重：{config.length_6_loss_weight}")
    print(f"最佳模型保存到：{model_path}")
    print(f"每轮训练集同条件评价：{config.evaluate_train}")
    print(f"逐 batch 稳定性诊断：{config.diagnose_batches}")
    print(f"BatchNorm 当前 batch 统计差分诊断：{config.diagnose_batchnorm}")
    print(f"BatchNorm 训练集统计重校准：{config.recalibrate_batchnorm}")
    print(
        f"自动降低学习率：{config.reduce_lr_on_plateau} | "
        f"factor={PLATEAU_FACTOR} | patience={PLATEAU_PATIENCE} | "
        f"min_lr={MIN_LEARNING_RATE}"
    )
    print("测试集不会在本训练入口中读取。")

    best_accuracy = -1.0
    best_character_error_rate = float("inf")
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        training_diagnostics = (
            run_training_epoch_with_diagnostics(
                model,
                train_loader,
                optimizer,
                device,
                length_6_loss_weight=config.length_6_loss_weight,
            )
            if config.diagnose_batches
            else None
        )
        training_loss = (
            training_diagnostics.average_loss
            if training_diagnostics is not None
            else run_training_epoch(
                model,
                train_loader,
                optimizer,
                device,
                length_6_loss_weight=config.length_6_loss_weight,
            )
        )
        uncalibrated_validation = (
            evaluate_model(model, validation_loader, device)
            if config.recalibrate_batchnorm
            else None
        )
        evaluation_model = (
            recalibrate_batchnorm_model(
                model,
                train_evaluation_loader,
                device,
            )
            if config.recalibrate_batchnorm and train_evaluation_loader is not None
            else model
        )
        train_evaluation = (
            evaluate_model(
                evaluation_model,
                train_evaluation_loader,
                device,
            )
            if train_evaluation_loader is not None
            else None
        )
        validation_loss, validation_result, _comparisons = evaluate_model(
            evaluation_model,
            validation_loader,
            device,
        )
        batchnorm_diagnostics = (
            (
                evaluate_model_with_batch_statistics(
                    model,
                    train_evaluation_loader,
                    device,
                ),
                evaluate_model_with_batch_statistics(
                    model,
                    validation_loader,
                    device,
                ),
            )
            if config.diagnose_batchnorm and train_evaluation_loader is not None
            else None
        )
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"学习率={current_learning_rate:.6f} | "
            f"train_loss={training_loss:.4f} | "
            f"validation_loss={validation_loss:.4f} | "
            f"验证整串准确率={validation_result.exact_match_accuracy:.2%} | "
            f"验证 CER={validation_result.character_error_rate:.2%}"
        )
        print(f"  验证按长度：{_format_per_length(validation_result)}")

        if uncalibrated_validation is not None:
            (
                uncalibrated_validation_loss,
                uncalibrated_validation_result,
                _uncalibrated_comparisons,
            ) = uncalibrated_validation
            recalibration_accuracy_delta = (
                validation_result.exact_match_accuracy
                - uncalibrated_validation_result.exact_match_accuracy
            )
            recalibration_cer_delta = (
                validation_result.character_error_rate
                - uncalibrated_validation_result.character_error_rate
            )
            print(
                "  BatchNorm校准："
                f"校准前验证 loss={uncalibrated_validation_loss:.4f} | "
                "准确率="
                f"{uncalibrated_validation_result.exact_match_accuracy:.2%} | "
                f"CER={uncalibrated_validation_result.character_error_rate:.2%} | "
                f"校准后准确率Δ={recalibration_accuracy_delta:+.2%} | "
                f"CERΔ={recalibration_cer_delta:+.2%}"
            )

        if training_diagnostics is not None:
            print(
                "  批次诊断："
                f"max_loss={training_diagnostics.max_batch_loss:.4f}"
                f"@{training_diagnostics.max_batch_loss_index} | "
                f"last_loss={training_diagnostics.last_batch_loss:.4f} | "
                f"max_grad_norm={training_diagnostics.max_gradient_norm:.4f}"
                f"@{training_diagnostics.max_gradient_norm_index} | "
                f"last_grad_norm={training_diagnostics.last_gradient_norm:.4f} | "
                f"nonfinite_loss={training_diagnostics.nonfinite_loss_count} | "
                "nonfinite_grad="
                f"{training_diagnostics.nonfinite_gradient_count}"
            )

        if train_evaluation is not None:
            train_evaluation_loss, train_result, _train_comparisons = train_evaluation
            accuracy_gap = (
                train_result.exact_match_accuracy
                - validation_result.exact_match_accuracy
            )
            cer_gap = (
                validation_result.character_error_rate
                - train_result.character_error_rate
            )
            print(
                f"  训练评价：loss={train_evaluation_loss:.4f} | "
                f"整串准确率={train_result.exact_match_accuracy:.2%} | "
                f"CER={train_result.character_error_rate:.2%}"
            )
            print(f"  训练按长度：{_format_per_length(train_result)}")
            print(f"  泛化差距：整串准确率={accuracy_gap:.2%} | CER={cer_gap:.2%}")

            if batchnorm_diagnostics is not None:
                batch_stat_train, batch_stat_validation = batchnorm_diagnostics
                (
                    batch_stat_train_loss,
                    batch_stat_train_result,
                    _batch_stat_train_comparisons,
                ) = batch_stat_train
                (
                    batch_stat_validation_loss,
                    batch_stat_validation_result,
                    _batch_stat_validation_comparisons,
                ) = batch_stat_validation
                train_accuracy_delta = (
                    batch_stat_train_result.exact_match_accuracy
                    - train_result.exact_match_accuracy
                )
                validation_accuracy_delta = (
                    batch_stat_validation_result.exact_match_accuracy
                    - validation_result.exact_match_accuracy
                )
                validation_cer_delta = (
                    batch_stat_validation_result.character_error_rate
                    - validation_result.character_error_rate
                )
                print(
                    "  BatchNorm差分："
                    f"batch-stat训练 loss={batch_stat_train_loss:.4f} | "
                    "准确率="
                    f"{batch_stat_train_result.exact_match_accuracy:.2%} "
                    f"(Δ={train_accuracy_delta:+.2%}) | "
                    f"batch-stat验证 loss={batch_stat_validation_loss:.4f} | "
                    "准确率="
                    f"{batch_stat_validation_result.exact_match_accuracy:.2%} "
                    f"(Δ={validation_accuracy_delta:+.2%}) | "
                    f"CER={batch_stat_validation_result.character_error_rate:.2%} "
                    f"(Δ={validation_cer_delta:+.2%})"
                )
                print(
                    "  batch-stat验证按长度："
                    f"{_format_per_length(batch_stat_validation_result)}"
                )

        if _is_better_validation_result(
            validation_result,
            validation_loss,
            best_accuracy=best_accuracy,
            best_character_error_rate=best_character_error_rate,
            best_loss=best_loss,
        ):
            best_accuracy = validation_result.exact_match_accuracy
            best_character_error_rate = validation_result.character_error_rate
            best_loss = validation_loss
            best_epoch = epoch
            _save_checkpoint(
                evaluation_model,
                model_path,
                config,
                epoch,
                validation_loss,
                validation_result,
            )
            print("  已更新最佳验证模型")

        if scheduler is not None:
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step(validation_loss)
            next_learning_rate = optimizer.param_groups[0]["lr"]
            if next_learning_rate < previous_learning_rate:
                print(
                    "  校准后验证 loss 停止改善，学习率调整："
                    f"{previous_learning_rate:.6f} -> {next_learning_rate:.6f}"
                )

    print(
        f"训练完成：最佳轮次={best_epoch} | "
        f"验证整串准确率={best_accuracy:.2%} | "
        f"验证 CER={best_character_error_rate:.2%} | "
        f"验证 loss={best_loss:.4f}"
    )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练第三阶段 CNN + CTC 正式基线")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--sequence-model",
        choices=SUPPORTED_SEQUENCE_MODELS,
        default="cnn",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=DEFAULT_LSTM_HIDDEN_SIZE,
    )
    parser.add_argument(
        "--sequence-width",
        type=int,
        choices=SUPPORTED_SEQUENCE_WIDTHS,
        default=DEFAULT_SEQUENCE_WIDTH,
    )
    parser.add_argument("--length-6-weight", type=float, default=1.0)
    parser.add_argument("--length-6-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluate-train",
        action="store_true",
        help="每轮结束后用 eval 模式重新评价完整训练集",
    )
    parser.add_argument(
        "--diagnose-batches",
        action="store_true",
        help="汇总每轮最大和最后 batch 的 loss 与梯度 L2 范数",
    )
    parser.add_argument(
        "--diagnose-batchnorm",
        action="store_true",
        help="用模型副本比较冻结运行统计与当前 batch 统计",
    )
    parser.add_argument(
        "--recalibrate-batchnorm",
        action="store_true",
        help="每轮用训练集重建模型副本的 BatchNorm 运行统计",
    )
    parser.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        help="校准后验证 loss 连续不改善时自动降低学习率",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    train_baseline(
        BaselineTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            sequence_model=args.sequence_model,
            lstm_hidden_size=args.lstm_hidden_size,
            sequence_width=args.sequence_width,
            length_6_weight=args.length_6_weight,
            length_6_loss_weight=args.length_6_loss_weight,
            seed=args.seed,
            device=args.device,
            evaluate_train=args.evaluate_train,
            diagnose_batches=args.diagnose_batches,
            diagnose_batchnorm=args.diagnose_batchnorm,
            recalibrate_batchnorm=args.recalibrate_batchnorm,
            reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        ),
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
