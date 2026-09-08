"""只在 V2 训练集的 10 张图片上做过拟合检查，不保存权重。

uv run python -m stage4_transformer_encoder.overfit_tiny
"""

import argparse
from pathlib import Path

import torch

from stage3_variable_length.data import ManifestCaptchaDataset
from stage3_variable_length.overfit_tiny import decode_batch
from stage3_variable_length.training import compute_ctc_loss, encode_ctc_targets

from .models import CaptchaTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs 必须大于 0")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    dataset = ManifestCaptchaDataset(Path("data/stage3_variable_length_v2/train"))
    if dataset.split != "train":
        raise ValueError("仅允许读取训练集")
    # 清单顺序固定，每种长度取前两张，不根据模型结果挑选样本。
    indices = []
    for length in range(2, 7):
        candidates = [
            i
            for i, record in enumerate(dataset.records)
            if len(str(record["label"])) == length
        ]
        if len(candidates) < 2:
            raise ValueError(f"长度 {length} 的训练样本不足两张")
        indices.extend(candidates[:2])
    samples = [dataset[index] for index in indices]
    images = torch.stack([image for image, _ in samples])
    labels = [label for _, label in samples]
    model = CaptchaTransformer(dataset.characters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    targets, lengths = encode_ctc_targets(labels, model.characters)
    print(
        f"CPU | seed=0 | lr=0.001 | batch=10 | 参数量={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    print(f"训练清单索引={indices} | 标签={labels}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(images)
        loss = compute_ctc_loss(logits, targets, lengths, model.blank_index)
        if not torch.isfinite(loss):
            raise RuntimeError("CTC loss 非有限值")
        optimizer.zero_grad()
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = decode_batch(
                model(images), model.characters, model.blank_index
            )
        correct = sum(a == b for a, b in zip(labels, predictions, strict=True))
        if epoch == 1 or epoch % 25 == 0 or correct == len(labels):
            print(
                f"epoch={epoch} | train-mode loss={loss.item():.4f} | eval-mode 正确={correct}/10",
                flush=True,
            )
        if correct == len(labels):
            print(list(zip(labels, predictions, strict=True)), flush=True)
            print("过拟合检查通过；未保存权重。", flush=True)
            return
    print(list(zip(labels, predictions, strict=True)), flush=True)
    raise SystemExit("达到轮数上限，尚未记住全部样本；未保存权重。")


if __name__ == "__main__":
    main()
