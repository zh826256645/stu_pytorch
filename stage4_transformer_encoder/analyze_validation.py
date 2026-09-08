"""只做验证集推理与错误统计，不训练、不修改 checkpoint。"""

import argparse
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stage3_variable_length.analyze_validation import (
    build_validation_analysis,
    print_analysis,
    write_validation_reports,
)
from stage3_variable_length.data import ManifestCaptchaDataset
from stage3_variable_length.train import choose_device, evaluate_model

from .models import CaptchaTransformer
from .train import MODEL_PATH, VALIDATION_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--decoder", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()
    if args.beam_width <= 0:
        parser.error("beam-width 必须大于 0")
    device = choose_device(args.device)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # 历史权重没有此字段，当时始终启用位置编码。
    model_config = {
        "use_position_encoding": True,
        "ffn_dim": 128,
        **checkpoint["model_config"],
    }
    expected_config = {
        "sequence_model": "transformer",
        "sequence_width": model_config["sequence_width"],
        "d_model": 64,
        "num_heads": 4,
        "dim_feedforward": model_config["dim_feedforward"],
        "ffn_dim": model_config["ffn_dim"],
        "num_layers": 1,
        "norm_first": model_config["norm_first"],
        "dropout": checkpoint["model_config"]["dropout"],
        "use_position_encoding": model_config["use_position_encoding"],
    }
    if model_config != expected_config:
        raise ValueError("checkpoint 结构不匹配当前教学模型")
    model = CaptchaTransformer(
        checkpoint["characters"],
        dropout=model_config["dropout"],
        use_position_encoding=model_config["use_position_encoding"],
        norm_first=model_config["norm_first"],
        ffn_dim=model_config["ffn_dim"],
        sequence_width=model_config["sequence_width"],
    ).to(device)
    if checkpoint["blank_index"] != model.blank_index:
        raise ValueError("checkpoint blank 索引不匹配")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = ManifestCaptchaDataset(VALIDATION_DIR)
    if dataset.split != "validation" or dataset.characters != model.characters:
        raise ValueError("需要字符表匹配的 validation 数据集")
    batch_size = int(checkpoint["training_config"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss, _, comparisons = evaluate_model(
        model, loader, device, decoder=args.decoder, beam_width=args.beam_width
    )
    summary, errors = build_validation_analysis(
        [
            {**record, "image_path": str(VALIDATION_DIR / record["file"])}
            for record in dataset.records
        ],
        [expected for expected, _ in comparisons],
        [predicted for _, predicted in comparisons],
        checkpoint_epoch=int(checkpoint["epoch"]),
        validation_loss=loss,
    )
    summary.update(
        checkpoint_path=str(MODEL_PATH),
        checkpoint_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        validation_directory=str(VALIDATION_DIR),
        device=str(device),
        decoder=args.decoder,
        beam_width=args.beam_width if args.decoder == "beam" else None,
    )
    report_directory = Path("stage4_transformer_encoder/evaluation_reports")
    if args.decoder == "beam":
        report_directory /= f"beam_{args.beam_width}"
    paths = write_validation_reports(summary, errors, report_directory)
    print(f"设备={device} | checkpoint epoch={checkpoint['epoch']} | loss={loss:.4f}")
    print_analysis(summary, *paths)


if __name__ == "__main__":
    main()
