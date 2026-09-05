"""在独立测试集上评估简单 CNN，并生成混淆矩阵和错误报告。

默认测试目录 ``data/independent_test`` 必须放入从未参与训练、模型选择或
超参数调整的新图片。若只是复查历史验证集，可显式传入 ``--data-dir data/test``
和 ``--allow-known-data``，但该结果不能视为独立测试准确率。
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from datasets import SUPPORTED_IMAGE_SUFFIXES, CaptchaData, alphabet

from .models import SUPPORTED_CLASSIFIER_HEADS, SUPPORTED_CONV_BLOCKS, SimpleCaptchaCNN
from .train import DEFAULT_MODEL_PATH, get_device

DEFAULT_TEST_DIR = Path("data/independent_test")
DEFAULT_REPORT_DIR = Path("stage2_simple_cnn/evaluation_reports")
DEFAULT_REFERENCE_DIRS = (Path("data/train"), Path("data/test"))


@dataclass(frozen=True)
class Mistake:
    """保存一张整图预测错误的详细信息。"""

    image: str
    expected: str
    predicted: str
    wrong_positions: tuple[int, ...]
    predicted_confidences: tuple[float, ...]


@dataclass(frozen=True)
class DuplicateImage:
    """保存独立测试集与历史数据之间的完全相同图片。"""

    test_image: str
    reference_image: str


@dataclass(frozen=True)
class EvaluationResult:
    """保存完整评估统计，供终端展示和文件报告共同使用。"""

    total_captchas: int
    correct_captchas: int
    correct_characters: int
    total_characters: int
    per_position_correct: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    mistakes: tuple[Mistake, ...]

    @property
    def exact_match_accuracy(self) -> float:
        return self.correct_captchas / self.total_captchas

    @property
    def character_accuracy(self) -> float:
        return self.correct_characters / self.total_characters

    @property
    def per_position_accuracy(self) -> tuple[float, ...]:
        return tuple(
            correct / self.total_captchas for correct in self.per_position_correct
        )


def list_image_files(directory: Path) -> list[Path]:
    """按文件名排序返回目录中的受支持图片。"""
    if not directory.is_dir():
        raise ValueError(f"图片目录不存在：{directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def calculate_sha256(path: Path) -> str:
    """流式计算图片哈希，避免一次把大文件全部读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_exact_duplicates(
    test_dir: Path,
    reference_dirs: list[Path],
) -> tuple[DuplicateImage, ...]:
    """查找测试图片与训练/验证图片之间的字节级重复。"""
    reference_by_hash: dict[str, Path] = {}
    for reference_dir in reference_dirs:
        if not reference_dir.is_dir():
            continue
        for image_path in list_image_files(reference_dir):
            reference_by_hash.setdefault(calculate_sha256(image_path), image_path)

    duplicates = []
    for test_image in list_image_files(test_dir):
        reference_image = reference_by_hash.get(calculate_sha256(test_image))
        if reference_image is not None:
            duplicates.append(
                DuplicateImage(
                    test_image=str(test_image),
                    reference_image=str(reference_image),
                )
            )
    return tuple(duplicates)


def build_evaluation_result(
    predicted_indices: torch.Tensor,
    expected_indices: torch.Tensor,
    predicted_confidences: torch.Tensor,
    image_paths: list[Path],
    *,
    class_names: str = alphabet,
) -> EvaluationResult:
    """根据类别编号构造准确率、混淆矩阵和逐图片错误记录。"""
    if predicted_indices.ndim != 2:
        raise ValueError("预测类别编号必须是 [图片数, 字符位置数]")
    if predicted_indices.shape != expected_indices.shape:
        raise ValueError("预测类别编号和真实类别编号形状必须一致")
    if predicted_indices.shape != predicted_confidences.shape:
        raise ValueError("预测置信度和预测类别编号形状必须一致")
    if predicted_indices.size(0) != len(image_paths):
        raise ValueError("图片路径数量必须与预测图片数量一致")
    if predicted_indices.size(0) == 0:
        raise ValueError("测试集不能为空")

    num_class = len(class_names)
    if predicted_indices.min().item() < 0 or expected_indices.min().item() < 0:
        raise ValueError("类别编号不能为负数")
    if predicted_indices.max().item() >= num_class:
        raise ValueError("预测类别编号超出字符集范围")
    if expected_indices.max().item() >= num_class:
        raise ValueError("真实类别编号超出字符集范围")

    matches = predicted_indices.eq(expected_indices)
    confusion_matrix = [[0 for _ in range(num_class)] for _ in range(num_class)]
    mistakes = []

    for image_index, image_path in enumerate(image_paths):
        expected_row = expected_indices[image_index].tolist()
        predicted_row = predicted_indices[image_index].tolist()
        confidence_row = predicted_confidences[image_index].tolist()

        for expected, predicted in zip(expected_row, predicted_row, strict=True):
            confusion_matrix[expected][predicted] += 1

        if not matches[image_index].all():
            mistakes.append(
                Mistake(
                    image=str(image_path),
                    expected="".join(class_names[index] for index in expected_row),
                    predicted="".join(class_names[index] for index in predicted_row),
                    wrong_positions=tuple(
                        position + 1
                        for position, is_correct in enumerate(
                            matches[image_index].tolist()
                        )
                        if not is_correct
                    ),
                    predicted_confidences=tuple(
                        float(value) for value in confidence_row
                    ),
                )
            )

    return EvaluationResult(
        total_captchas=predicted_indices.size(0),
        correct_captchas=matches.all(dim=1).sum().item(),
        correct_characters=matches.sum().item(),
        total_characters=matches.numel(),
        per_position_correct=tuple(matches.sum(dim=0).tolist()),
        confusion_matrix=tuple(tuple(row) for row in confusion_matrix),
        mistakes=tuple(mistakes),
    )


def evaluate_model(
    model: SimpleCaptchaCNN,
    data_loader: DataLoader,
    image_paths: list[Path],
    device: torch.device,
) -> EvaluationResult:
    """运行模型推理并汇总完整测试结果。"""
    model.eval()
    predicted_batches = []
    expected_batches = []
    confidence_batches = []

    with torch.inference_mode():
        for images, targets in data_loader:
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=-1).cpu()
            predicted_confidences, predicted_indices = probabilities.max(dim=-1)
            expected_indices = targets.view(
                -1,
                model.num_char,
                model.num_class,
            ).argmax(dim=-1)

            predicted_batches.append(predicted_indices)
            expected_batches.append(expected_indices)
            confidence_batches.append(predicted_confidences)

    if not predicted_batches:
        raise ValueError("测试集不能为空")

    return build_evaluation_result(
        torch.cat(predicted_batches),
        torch.cat(expected_batches),
        torch.cat(confidence_batches),
        image_paths,
    )


def write_confusion_csv(
    result: EvaluationResult,
    output_path: Path,
    *,
    class_names: str = alphabet,
) -> None:
    """写出完整字符级混淆矩阵 CSV。"""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["expected\\predicted", *class_names])
        for class_name, row in zip(
            class_names,
            result.confusion_matrix,
            strict=True,
        ):
            writer.writerow([class_name, *row])


def write_errors_csv(result: EvaluationResult, output_path: Path) -> None:
    """写出所有整图错误及各位置预测置信度。"""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image",
                "expected",
                "predicted",
                "wrong_positions",
                "predicted_confidences",
            ]
        )
        for mistake in result.mistakes:
            writer.writerow(
                [
                    mistake.image,
                    mistake.expected,
                    mistake.predicted,
                    " ".join(str(value) for value in mistake.wrong_positions),
                    " ".join(
                        f"{confidence:.6f}"
                        for confidence in mistake.predicted_confidences
                    ),
                ]
            )


def write_confusion_plot(
    result: EvaluationResult,
    output_path: Path,
    *,
    class_names: str = alphabet,
) -> None:
    """只绘制测试中出现过的类别，使小数据集矩阵保持可读。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    active_indices = [
        index
        for index in range(len(class_names))
        if sum(result.confusion_matrix[index]) > 0
        or sum(row[index] for row in result.confusion_matrix) > 0
    ]
    active_matrix = [
        [result.confusion_matrix[row][column] for column in active_indices]
        for row in active_indices
    ]
    active_names = [class_names[index] for index in active_indices]

    figure_size = max(7, len(active_names) * 0.45)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(active_matrix, cmap="Blues")
    axis.set_title("Character Confusion Matrix")
    axis.set_xlabel("Predicted character")
    axis.set_ylabel("Expected character")
    axis.set_xticks(range(len(active_names)), active_names)
    axis.set_yticks(range(len(active_names)), active_names)

    for row, expected_index in enumerate(active_indices):
        for column, predicted_index in enumerate(active_indices):
            value = result.confusion_matrix[expected_index][predicted_index]
            if value > 0 and row != column:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=9,
                )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_reports(
    result: EvaluationResult,
    output_dir: Path,
    duplicates: tuple[DuplicateImage, ...],
    *,
    class_names: str = alphabet,
) -> None:
    """生成机器可读指标、混淆矩阵和逐图片错误报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "total_captchas": result.total_captchas,
        "correct_captchas": result.correct_captchas,
        "exact_match_accuracy": result.exact_match_accuracy,
        "correct_characters": result.correct_characters,
        "total_characters": result.total_characters,
        "character_accuracy": result.character_accuracy,
        "per_position_accuracy": result.per_position_accuracy,
        "error_count": len(result.mistakes),
        "exact_duplicate_count": len(duplicates),
        "exact_duplicates": [asdict(duplicate) for duplicate in duplicates],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_confusion_csv(
        result,
        output_dir / "confusion_matrix.csv",
        class_names=class_names,
    )
    write_errors_csv(result, output_dir / "errors.csv")
    write_confusion_plot(
        result,
        output_dir / "confusion_matrix.png",
        class_names=class_names,
    )


def print_summary(result: EvaluationResult, output_dir: Path) -> None:
    """在终端打印最重要的准确率和非对角混淆。"""
    print(
        "整图准确率："
        f"{result.correct_captchas}/{result.total_captchas} "
        f"= {result.exact_match_accuracy:.2%}"
    )
    print(
        "单字符准确率："
        f"{result.correct_characters}/{result.total_characters} "
        f"= {result.character_accuracy:.2%}"
    )
    print(
        "各位置准确率："
        + " | ".join(
            f"位置 {position}: {accuracy:.2%}"
            for position, accuracy in enumerate(
                result.per_position_accuracy,
                start=1,
            )
        )
    )
    print(f"整图错误数：{len(result.mistakes)}")

    confusions = Counter()
    for expected, row in enumerate(result.confusion_matrix):
        for predicted, count in enumerate(row):
            if expected != predicted and count:
                confusions[(alphabet[expected], alphabet[predicted])] += count

    if confusions:
        print("主要字符混淆：")
        for (expected, predicted), count in confusions.most_common(10):
            print(f"  {expected} -> {predicted}: {count}")
    else:
        print("主要字符混淆：无")
    print(f"评估报告：{output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在独立测试集上评估简单 CNN")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help=f"独立测试图片目录，默认 {DEFAULT_TEST_DIR}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"模型权重路径，默认 {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"评估报告目录，默认 {DEFAULT_REPORT_DIR}",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--conv-blocks",
        type=int,
        choices=SUPPORTED_CONV_BLOCKS,
        default=4,
    )
    parser.add_argument("--bottleneck-channels", type=int, default=64)
    parser.add_argument(
        "--classifier-head",
        choices=SUPPORTED_CLASSIFIER_HEADS,
        default="position",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        action="append",
        default=None,
        help="用于重复检查的历史数据目录；可重复传入",
    )
    parser.add_argument(
        "--allow-known-data",
        action="store_true",
        help="允许评估历史训练/验证目录或重复图片，仅用于排查，不算独立测试",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    image_paths = list_image_files(args.data_dir)
    if not image_paths:
        raise ValueError(
            f"独立测试目录没有图片：{args.data_dir}。请放入从未参与训练和模型选择的"
            "新图片，文件名需为 4 位真实标签，例如 2n6m.png。"
        )

    reference_dirs = args.reference_dir or list(DEFAULT_REFERENCE_DIRS)
    resolved_data_dir = args.data_dir.resolve()
    known_directory = any(
        reference_dir.exists() and reference_dir.resolve() == resolved_data_dir
        for reference_dir in reference_dirs
    )
    duplicates = find_exact_duplicates(args.data_dir, reference_dirs)
    if (known_directory or duplicates) and not args.allow_known_data:
        raise ValueError(
            "测试目录属于历史数据，或包含与训练/验证集完全相同的图片。"
            "请更换真正未见过的数据；若只想排查历史数据，可添加 "
            "--allow-known-data。"
        )

    dataset = CaptchaData(str(args.data_dir), transform=ToTensor())
    if len(dataset) != len(image_paths):
        raise ValueError("测试目录只能包含受支持的图片文件")
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()
    model = SimpleCaptchaCNN(
        num_conv_blocks=args.conv_blocks,
        bottleneck_channels=args.bottleneck_channels,
        classifier_head=args.classifier_head,
        dropout=args.dropout,
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            "权重与指定模型结构不匹配，请检查卷积块、瓶颈通道和分类头参数"
        ) from error

    print(f"设备：{device} | 测试图片：{len(dataset)} | 权重：{args.model_path}")
    if args.allow_known_data:
        print("警告：当前允许使用历史数据，本次结果不能视为独立测试准确率。")
    print(f"与历史数据完全重复的图片：{len(duplicates)}")

    result = evaluate_model(model, data_loader, image_paths, device)
    write_reports(result, args.output_dir, duplicates)
    print_summary(result, args.output_dir)
    if args.allow_known_data:
        print("本次只用于排查历史数据，不记录为独立测试结果。")
    else:
        print("请锁定本次结果，不要根据独立测试集错误继续调整模型或超参数。")


if __name__ == "__main__":
    main()
