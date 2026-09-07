"""让第三阶段 CNN + CTC 记住极小干净数据集的诊断实验。

运行方式：
    uv run python -m stage3_variable_length.overfit_tiny
"""

import argparse
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from .data import TinyCaptchaDataset
from .decoding import decode_ctc_path
from .models import DEFAULT_CHARACTERS, VariableLengthCaptchaCNN
from .training import compute_ctc_loss, encode_ctc_targets

TINY_LABELS = (
    "aa",
    "27",
    "a7b",
    "33m",
    "b8x2",
    "pwaa",
    "xxy23",
    "7mnpw",
    "33aabb",
    "2g7xmy",
)


def choose_device(requested: str) -> torch.device:
    """根据命令行参数选择训练设备。"""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode_batch(
    logits: torch.Tensor,
    characters: str,
    blank_index: int,
) -> list[str]:
    """对 batch-first logits 执行逐时间步 argmax 和 CTC 路径折叠。"""
    paths = logits.argmax(dim=2).detach().cpu().tolist()
    return [decode_ctc_path(path, characters, blank_index) for path in paths]


def evaluate_training_set(
    model: VariableLengthCaptchaCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[tuple[str, str]]]:
    """返回极小训练集的整串准确率以及目标和预测对。"""
    model.eval()
    comparisons: list[tuple[str, str]] = []

    with torch.inference_mode():
        for images, labels in data_loader:
            logits = model(images.to(device))
            predictions = decode_batch(
                logits,
                model.characters,
                model.blank_index,
            )
            comparisons.extend(zip(labels, predictions, strict=True))

    correct = sum(expected == predicted for expected, predicted in comparisons)
    return correct / len(comparisons), comparisons


def run_overfit_experiment(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    report_every: int,
    seed: int,
    device_name: str,
) -> bool:
    """运行极小数据集诊断；达到 100% 整串准确率时提前停止。"""
    if epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if report_every <= 0:
        raise ValueError("report_every 必须大于 0")

    torch.manual_seed(seed)
    device = choose_device(device_name)
    dataset = TinyCaptchaDataset(TINY_LABELS, seed=seed)
    training_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    evaluation_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
    )

    model = VariableLengthCaptchaCNN(characters=DEFAULT_CHARACTERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"使用设备：{device}")
    print(f"固定标签：{list(TINY_LABELS)}")
    print(
        f"样本数：{len(dataset)} | batch size：{min(batch_size, len(dataset))} "
        f"| 最大轮数：{epochs} | 学习率：{learning_rate}"
    )

    reached_full_accuracy = False
    final_comparisons: list[tuple[str, str]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in training_loader:
            targets, target_lengths = encode_ctc_targets(
                labels,
                model.characters,
            )
            logits = model(images.to(device))
            loss = compute_ctc_loss(
                logits,
                targets,
                target_lengths,
                blank_index=model.blank_index,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = images.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        exact_match_accuracy, final_comparisons = evaluate_training_set(
            model,
            evaluation_loader,
            device,
        )
        average_loss = total_loss / total_samples
        if epoch == 1 or epoch % report_every == 0 or exact_match_accuracy == 1.0:
            print(
                f"Epoch {epoch:03d} | loss={average_loss:.4f} "
                f"| 整串准确率={exact_match_accuracy:.2%}"
            )

        if exact_match_accuracy == 1.0:
            reached_full_accuracy = True
            break

    print("\n最终逐样本结果：")
    for expected, predicted in final_comparisons:
        marker = "✓" if expected == predicted else "✗"
        print(f"{marker} 目标={expected!r:<10} 预测={predicted!r}")

    if reached_full_accuracy:
        print("\n诊断通过：模型已经记住全部固定样本。")
    else:
        print("\n诊断未通过：在最大轮数内没有达到 100% 整串准确率。")
    return reached_full_accuracy


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="让 CNN + CTC 记住 10 张干净图片，检查训练链路",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    succeeded = run_overfit_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        report_every=args.report_every,
        seed=args.seed,
        device_name=args.device,
    )
    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
