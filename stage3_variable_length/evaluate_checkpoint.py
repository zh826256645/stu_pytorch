"""同条件评估第三阶段最佳 checkpoint 的训练集与验证集。

运行方式：
    uv run python -m stage3_variable_length.evaluate_checkpoint
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ManifestCaptchaDataset
from .metrics import SequenceEvaluation
from .models import (
    DEFAULT_LSTM_HIDDEN_SIZE,
    DEFAULT_SEQUENCE_WIDTH,
    VariableLengthCaptchaCNN,
)
from .train import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_DIR,
    DEFAULT_VALIDATION_DIR,
    SUPPORTED_DECODERS,
    choose_device,
    evaluate_model,
)


@dataclass(frozen=True)
class SplitCheckpointEvaluation:
    """最佳 checkpoint 在一个数据集合上的只读评价结果。"""

    split: str
    loss: float
    metrics: SequenceEvaluation


def load_checkpoint_model(
    model_path: Path,
    device: torch.device,
) -> tuple[VariableLengthCaptchaCNN, dict[str, object]]:
    """加载第三阶段 checkpoint，校验字符表与 blank 后切换到 eval。"""
    if not model_path.is_file():
        raise ValueError(f"checkpoint 不存在：{model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("不支持的 checkpoint 版本")
    characters = checkpoint.get("characters")
    blank_index = checkpoint.get("blank_index")
    if not isinstance(characters, str) or not characters:
        raise ValueError("checkpoint 缺少有效字符表")
    if blank_index != len(characters):
        raise ValueError("checkpoint 的 blank_index 与字符表不一致")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_state_dict")

    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 的 model_config 无效")
    sequence_model = model_config.get("sequence_model", "cnn")
    lstm_hidden_size = model_config.get(
        "lstm_hidden_size",
        DEFAULT_LSTM_HIDDEN_SIZE,
    )
    sequence_width = model_config.get(
        "sequence_width",
        DEFAULT_SEQUENCE_WIDTH,
    )
    if not isinstance(sequence_model, str):
        raise ValueError("checkpoint 的 sequence_model 无效")
    if not isinstance(lstm_hidden_size, int):
        raise ValueError("checkpoint 的 lstm_hidden_size 无效")
    if not isinstance(sequence_width, int):
        raise ValueError("checkpoint 的 sequence_width 无效")

    model = VariableLengthCaptchaCNN(
        characters=characters,
        sequence_model=sequence_model,
        lstm_hidden_size=lstm_hidden_size,
        sequence_width=sequence_width,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    *,
    model_path: Path,
    train_dir: Path,
    validation_dir: Path,
    batch_size: int,
    device_name: str,
    decoder: str = "greedy",
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> tuple[dict[str, SplitCheckpointEvaluation], dict[str, object]]:
    """用同一最佳模型评估训练和验证数据，不读取测试集。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if decoder not in SUPPORTED_DECODERS:
        supported = ", ".join(SUPPORTED_DECODERS)
        raise ValueError(f"decoder 必须是以下值之一：{supported}")
    if beam_width <= 0:
        raise ValueError("beam_width 必须大于 0")

    device = choose_device(device_name)
    model, checkpoint = load_checkpoint_model(model_path, device)
    datasets = {
        "train": ManifestCaptchaDataset(train_dir),
        "validation": ManifestCaptchaDataset(validation_dir),
    }
    evaluations = {}
    for split, dataset in datasets.items():
        if dataset.split != split:
            raise ValueError(f"{split} 目录中的 manifest split 不一致")
        if dataset.characters != model.characters:
            raise ValueError(f"{split} 数据字符表与 checkpoint 不一致")

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        loss, metrics, _comparisons = evaluate_model(
            model,
            data_loader,
            device,
            decoder=decoder,
            beam_width=beam_width,
        )
        evaluations[split] = SplitCheckpointEvaluation(
            split=split,
            loss=loss,
            metrics=metrics,
        )

    return evaluations, checkpoint


def _format_per_length(metrics: SequenceEvaluation) -> str:
    return " | ".join(
        f"{length}位={result.accuracy:.2%}"
        for length, result in metrics.per_length.items()
    )


def print_evaluations(
    evaluations: dict[str, SplitCheckpointEvaluation],
    checkpoint: dict[str, object],
) -> None:
    """打印训练、验证指标以及两者的泛化差距。"""
    print(
        f"checkpoint epoch：{checkpoint.get('epoch')} | "
        f"保存时验证准确率："
        f"{float(checkpoint.get('validation_exact_match_accuracy', 0)):.2%}"
    )
    for split in ("train", "validation"):
        evaluation = evaluations[split]
        metrics = evaluation.metrics
        print(
            f"{split}: loss={evaluation.loss:.4f} | "
            f"整串准确率={metrics.exact_match_accuracy:.2%} | "
            f"CER={metrics.character_error_rate:.2%}"
        )
        print(f"  按长度：{_format_per_length(metrics)}")

    train_metrics = evaluations["train"].metrics
    validation_metrics = evaluations["validation"].metrics
    accuracy_gap = (
        train_metrics.exact_match_accuracy - validation_metrics.exact_match_accuracy
    )
    cer_gap = (
        validation_metrics.character_error_rate - train_metrics.character_error_rate
    )
    print(f"泛化差距：整串准确率 {accuracy_gap:.2%} | CER {cer_gap:.2%}")
    print("测试集未读取。")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用最佳 checkpoint 同条件评估训练集和验证集",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--decoder",
        choices=SUPPORTED_DECODERS,
        default="greedy",
    )
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    evaluations, checkpoint = evaluate_checkpoint(
        model_path=args.model_path,
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        decoder=args.decoder,
        beam_width=args.beam_width,
    )
    print(
        f"解码器：{args.decoder}"
        + (f" | beam width：{args.beam_width}" if args.decoder == "beam" else "")
    )
    print_evaluations(evaluations, checkpoint)


if __name__ == "__main__":
    main()
