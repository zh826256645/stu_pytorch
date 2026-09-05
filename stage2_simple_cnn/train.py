import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import ColorJitter, Compose, RandomAffine, ToTensor

from datasets import CaptchaData, alphabet
from training_config import TrainingConfig
from training_curves import TrainingCurvePlotter
from training_metrics import calculate_accuracy_counts

from .data_loading import MixedBatchSampler
from .models import (
    DEFAULT_DROPOUT,
    SUPPORTED_CLASSIFIER_HEADS,
    SUPPORTED_CONV_BLOCKS,
    SimpleCaptchaCNN,
)

# 当前验证码固定为 4 个字符。
NUM_CHAR = 4
# 字符集为 0-9 和 a-z，共 36 类。
NUM_CLASS = len(alphabet)
DEFAULT_MODEL_PATH = Path("stage2_simple_cnn/checkpoints/model.pth")
DEFAULT_TRAIN_DIR = Path("data/train")
DEFAULT_VALIDATION_DIR = Path("data/test")
DEFAULT_SEED = 0

# 两个阶段共用 TrainingConfig 类，但第二阶段使用适合简单 CNN 的默认值。
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    epochs=10,
    batch_size=16,
    learning_rate=0.001,
)
# 验证 loss 连续若干轮没有改善时，把学习率降低到原来的 30%。
PLATEAU_FACTOR = 0.3
PLATEAU_PATIENCE = 3
MIN_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_LABEL_SMOOTHING = 0.0


def set_random_seed(seed: int) -> None:
    """设置 Python 和 PyTorch 随机种子。"""
    if seed < 0:
        raise ValueError("seed 必须大于等于 0")
    random.seed(seed)
    torch.manual_seed(seed)


def is_better_checkpoint(
    *,
    accuracy: float,
    loss: float,
    best_accuracy: float,
    best_loss: float,
) -> bool:
    """整图准确率优先；准确率相同时选择验证 loss 更低的模型。"""
    return accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss)


def get_device() -> torch.device:
    """按照 CUDA、Apple MPS、CPU 的顺序选择训练设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_image_transform(*, augment: bool, color_jitter: bool = False) -> Compose:
    """构造图片预处理；训练时可选择几何和颜色增强。"""
    transforms = []
    if augment:
        # 验证码不能翻转，也不宜大幅裁剪。这里只做很小的旋转、平移、缩放和错切，
        # 模拟字符整体位置和书写角度变化，同时保持 4 个字符的顺序不变。
        transforms.append(
            RandomAffine(
                degrees=3,
                translate=(0.03, 0.05),
                scale=(0.95, 1.05),
                shear=3,
                fill=255,
            )
        )
    if color_jitter:
        # 颜色不是验证码标签的一部分。轻微扰动亮度、对比度和色彩，帮助模型
        # 更多关注字符形状，同时避免过强变化掩盖原本较细的字符笔画。
        transforms.append(
            ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            )
        )
    transforms.append(ToTensor())
    return Compose(transforms)


def build_train_loader(
    real_dataset: Dataset,
    synthetic_dataset: Dataset | None,
    *,
    batch_size: int,
    synthetic_ratio: float,
    seed: int,
) -> tuple[DataLoader, float]:
    """构造真实数据基线或固定来源比例的混合训练加载器。"""
    if not 0 <= synthetic_ratio < 1:
        raise ValueError("synthetic_ratio 必须大于等于 0 且小于 1")
    generator = torch.Generator().manual_seed(seed)

    if synthetic_ratio == 0:
        if synthetic_dataset is not None:
            raise ValueError("synthetic_ratio 为 0 时不应提供 synthetic_dataset")
        return (
            DataLoader(
                real_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            ),
            0.0,
        )

    if synthetic_dataset is None:
        raise ValueError("synthetic_ratio 大于 0 时必须提供 synthetic_dataset")

    combined_dataset = ConcatDataset([real_dataset, synthetic_dataset])
    batch_sampler = MixedBatchSampler(
        real_size=len(real_dataset),
        synthetic_size=len(synthetic_dataset),
        batch_size=batch_size,
        synthetic_ratio=synthetic_ratio,
        generator=generator,
    )
    return (
        DataLoader(
            combined_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        ),
        batch_sampler.actual_synthetic_ratio,
    )


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """构造优化器；启用权重衰减时使用解耦衰减的 AdamW。"""
    if weight_decay < 0:
        raise ValueError("weight_decay 必须大于等于 0")
    if weight_decay == 0:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_criterion(*, label_smoothing: float) -> nn.CrossEntropyLoss:
    """构造交叉熵损失，并限制标签平滑系数处于有效范围。"""
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing 必须大于等于 0 且小于 1")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """验证 loss 停止改善时自动降低学习率。"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )


def one_hot_to_class_indices(target: torch.Tensor) -> torch.Tensor:
    """把现有数据集的 one-hot 标签转换成类别编号。

    CaptchaData 返回的单个标签形状是 [144]，也就是 4 组 36 维
    one-hot 向量。CrossEntropyLoss 需要的是类别编号，因此转换为
    [批量大小, 4]。
    """
    target = target.view(-1, NUM_CHAR, NUM_CLASS)
    return target.argmax(dim=2)


def calculate_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """计算 4 个字符位置的平均交叉熵损失。"""
    target_indices = one_hot_to_class_indices(target)

    # CrossEntropyLoss 接收：
    #   预测：[样本数量, 类别数量]
    #   标签：[样本数量]
    # 因此把 B 张图片的 4 个位置合并为 B * 4 个分类样本。
    return criterion(
        logits.reshape(-1, NUM_CLASS),
        target_indices.reshape(-1),
    )


def run_epoch(
    model: SimpleCaptchaCNN,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """运行一轮训练或验证，并返回 loss、整图和单字符准确率。"""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct_captchas = 0
    total_correct_characters = 0
    total_images = 0

    # 训练时需要梯度；验证时关闭梯度可以节省内存和计算量。
    with torch.set_grad_enabled(is_training):
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播：让模型根据图片产生预测结果。
            logits = model(images)
            loss = calculate_loss(logits, targets, criterion)

            if is_training:
                # 清空上一批数据留下的梯度。
                optimizer.zero_grad()
                # 反向传播，计算每个参数的梯度。
                loss.backward()
                # 根据梯度更新模型参数。
                optimizer.step()

            current_batch_size = images.size(0)
            accuracy_counts = calculate_accuracy_counts(
                logits,
                targets,
                num_char=NUM_CHAR,
                num_class=NUM_CLASS,
            )
            total_images += current_batch_size
            total_loss += loss.item() * current_batch_size
            total_correct_captchas += accuracy_counts.correct_captchas
            total_correct_characters += accuracy_counts.correct_characters

    return (
        total_loss / total_images,
        total_correct_captchas / total_images,
        total_correct_characters / (total_images * NUM_CHAR),
    )


def train(
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    model_path: Path = DEFAULT_MODEL_PATH,
    *,
    num_conv_blocks: int = 3,
    bottleneck_channels: int | None = None,
    classifier_head: str = "flatten",
    dropout: float = DEFAULT_DROPOUT,
    seed: int = DEFAULT_SEED,
    augment: bool = False,
    color_jitter: bool = False,
    evaluate_clean_train: bool = False,
    reduce_lr_on_plateau: bool = False,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    synthetic_train_dir: Path | None = None,
    synthetic_ratio: float = 0.0,
    synthetic_validation_dir: Path | None = None,
) -> None:
    """训练指定结构的简单 CNN，并保存验证集最佳模型。"""
    # TrainingConfig 创建时已经统一检查 epochs、batch_size 和学习率。
    if not 0 <= synthetic_ratio < 1:
        raise ValueError("synthetic_ratio 必须大于等于 0 且小于 1")
    if synthetic_ratio > 0 and synthetic_train_dir is None:
        raise ValueError("启用合成训练比例时必须提供 synthetic_train_dir")
    if synthetic_ratio == 0 and synthetic_train_dir is not None:
        raise ValueError("提供 synthetic_train_dir 时 synthetic_ratio 必须大于 0")

    set_random_seed(seed)
    device = get_device()
    bottleneck_description = (
        "关闭" if bottleneck_channels is None else str(bottleneck_channels)
    )
    optimizer_name = "AdamW" if weight_decay > 0 else "Adam"
    print(
        f"使用设备：{device} | 卷积块：{num_conv_blocks} | "
        f"瓶颈通道：{bottleneck_description} | 分类头：{classifier_head} | "
        f"Dropout：{dropout} | 训练轮数：{config.epochs} | "
        f"批量大小：{config.batch_size} | "
        f"初始学习率：{config.learning_rate} | 随机种子：{seed} | "
        f"优化器：{optimizer_name} | 权重衰减：{weight_decay} | "
        f"标签平滑：{label_smoothing} | "
        f"学习率调度：{reduce_lr_on_plateau} | 几何增强：{augment} | "
        f"颜色增强：{color_jitter} | 干净训练评估：{evaluate_clean_train} | "
        f"合成训练目标比例：{synthetic_ratio:.2%} | "
        f"显示曲线：{config.plot_curves}"
    )
    print(
        f"真实训练集：{train_dir} | 原始验证集：{validation_dir} | "
        f"合成训练集：{synthetic_train_dir or '关闭'} | "
        f"合成验证集：{synthetic_validation_dir or '关闭'}"
    )
    print(f"权重保存路径：{model_path}")

    # plot_curves=False 时，这个共用模块不会导入或打开 Matplotlib。
    curve_plotter = TrainingCurvePlotter(
        enabled=config.plot_curves,
        title="Simple CNN Training Curves",
    )

    # 只增强训练集，验证集始终只执行 ToTensor，确保不同实验的指标可比较。
    train_transform = build_image_transform(
        augment=augment,
        color_jitter=color_jitter,
    )
    validation_transform = build_image_transform(
        augment=False,
        color_jitter=False,
    )
    train_dataset = CaptchaData(str(train_dir), transform=train_transform)
    synthetic_train_dataset = (
        CaptchaData(str(synthetic_train_dir), transform=train_transform)
        if synthetic_train_dir is not None
        else None
    )
    clean_train_dataset = (
        CaptchaData(str(train_dir), transform=validation_transform)
        if evaluate_clean_train
        else None
    )
    validation_dataset = CaptchaData(
        str(validation_dir),
        transform=validation_transform,
    )
    synthetic_validation_dataset = (
        CaptchaData(
            str(synthetic_validation_dir),
            transform=validation_transform,
        )
        if synthetic_validation_dir is not None
        else None
    )

    train_loader, actual_synthetic_ratio = build_train_loader(
        train_dataset,
        synthetic_train_dataset,
        batch_size=config.batch_size,
        synthetic_ratio=synthetic_ratio,
        seed=seed,
    )
    print(
        f"每轮训练批次数：{len(train_loader)} | "
        f"实际批次合成比例：{actual_synthetic_ratio:.2%}"
    )
    clean_train_loader = (
        DataLoader(
            clean_train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if clean_train_dataset is not None
        else None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    synthetic_validation_loader = (
        DataLoader(
            synthetic_validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if synthetic_validation_dataset is not None
        else None
    )

    model = SimpleCaptchaCNN(
        num_class=NUM_CLASS,
        num_char=NUM_CHAR,
        num_conv_blocks=num_conv_blocks,
        bottleneck_channels=bottleneck_channels,
        classifier_head=classifier_head,
        dropout=dropout,
    ).to(device)

    # 每个字符位置都是一个 36 分类问题。标签平滑大于 0 时，
    # 避免模型把训练标签拟合成过度自信的硬概率分布。
    criterion = build_criterion(label_smoothing=label_smoothing)
    # 权重衰减为 0 时保留历史 Adam 基线；大于 0 时使用 AdamW，
    # 将权重衰减与梯度更新解耦，便于控制变量比较泛化效果。
    optimizer = build_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer) if reduce_lr_on_plateau else None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_synthetic_accuracy: float | None = None

    for epoch in range(1, config.epochs + 1):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy, train_character_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        clean_train_metrics = (
            run_epoch(
                model,
                clean_train_loader,
                criterion,
                device,
            )
            if clean_train_loader is not None
            else None
        )
        validation_loss, validation_accuracy, validation_character_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        synthetic_validation_metrics = (
            run_epoch(
                model,
                synthetic_validation_loader,
                criterion,
                device,
            )
            if synthetic_validation_loader is not None
            else None
        )

        print(
            f"第 {epoch:02d}/{config.epochs} 轮 | "
            f"学习率: {current_learning_rate:.6f} | "
            f"训练 loss: {train_loss:.4f} | "
            f"训练整图准确率: {train_accuracy:.2%} | "
            f"训练单字符准确率: {train_character_accuracy:.2%} | "
            f"原始验证 loss: {validation_loss:.4f} | "
            f"原始验证整图准确率: {validation_accuracy:.2%} | "
            f"原始验证单字符准确率: {validation_character_accuracy:.2%}"
        )
        if synthetic_validation_metrics is not None:
            (
                synthetic_validation_loss,
                synthetic_validation_accuracy,
                synthetic_validation_character_accuracy,
            ) = synthetic_validation_metrics
            print(
                f"第 {epoch:02d}/{config.epochs} 轮合成验证 | "
                f"loss: {synthetic_validation_loss:.4f} | "
                f"整图准确率: {synthetic_validation_accuracy:.2%} | "
                "单字符准确率: "
                f"{synthetic_validation_character_accuracy:.2%}"
            )

        if clean_train_metrics is not None:
            (
                clean_train_loss,
                clean_train_accuracy,
                clean_train_character_accuracy,
            ) = clean_train_metrics
            print(
                f"第 {epoch:02d}/{config.epochs} 轮干净训练评估 | "
                f"loss: {clean_train_loss:.4f} | "
                f"整图准确率: {clean_train_accuracy:.2%} | "
                f"单字符准确率: {clean_train_character_accuracy:.2%}"
            )
        else:
            clean_train_loss = train_loss
            clean_train_accuracy = train_accuracy
            clean_train_character_accuracy = train_character_accuracy

        # 启用干净训练评估时，曲线使用与验证集相同条件下的训练集指标。
        curve_plotter.update(
            epoch=epoch,
            train_loss=clean_train_loss,
            validation_loss=validation_loss,
            train_accuracy=clean_train_accuracy,
            validation_accuracy=validation_accuracy,
            train_character_accuracy=clean_train_character_accuracy,
            validation_character_accuracy=validation_character_accuracy,
        )

        if scheduler is not None:
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step(validation_loss)
            next_learning_rate = optimizer.param_groups[0]["lr"]
            if next_learning_rate < previous_learning_rate:
                print(
                    "验证 loss 停止改善，学习率调整："
                    f"{previous_learning_rate:.6f} -> {next_learning_rate:.6f}"
                )

        # 优先保存验证集整图准确率更高的权重；准确率相同时选择 loss 更低者。
        if is_better_checkpoint(
            accuracy=validation_accuracy,
            loss=validation_loss,
            best_accuracy=best_accuracy,
            best_loss=best_loss,
        ):
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_synthetic_accuracy = (
                synthetic_validation_metrics[1]
                if synthetic_validation_metrics is not None
                else None
            )
            torch.save(model.state_dict(), model_path)
            print(f"已保存当前最佳模型：{model_path}")

    # 启用 --plot-curves 时，训练结束后等待用户关闭曲线窗口。
    curve_plotter.show()
    summary = (
        f"训练完成，最佳原始验证整图准确率：{best_accuracy:.2%} | "
        f"对应验证 loss：{best_loss:.4f}"
    )
    if best_synthetic_accuracy is not None:
        summary += f" | 同轮合成验证整图准确率：{best_synthetic_accuracy:.2%}"
    print(f"{summary} | 权重：{model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练第一个简单验证码 CNN")
    # --epochs、--batch-size、--lr、--plot-curves 由共用参数类统一添加。
    TrainingConfig.add_arguments(parser, DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="最佳模型权重保存路径；默认覆盖第二阶段统一权重",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help=f"真实训练集目录，默认 {DEFAULT_TRAIN_DIR}",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help=f"原始分布验证集目录，默认 {DEFAULT_VALIDATION_DIR}",
    )
    parser.add_argument(
        "--synthetic-train-dir",
        type=Path,
        default=None,
        help="合成训练集目录；与 --synthetic-ratio 一起启用",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.0,
        help="每个训练批次中的合成样本目标比例，范围 [0, 1)，默认 0",
    )
    parser.add_argument(
        "--synthetic-validation-dir",
        type=Path,
        default=None,
        help="每轮额外评估的合成验证集目录",
    )
    parser.add_argument(
        "--conv-blocks",
        type=int,
        choices=SUPPORTED_CONV_BLOCKS,
        default=3,
        help="卷积块数量；默认 3，设置为 4 可进行深度消融实验",
    )
    parser.add_argument(
        "--bottleneck-channels",
        type=int,
        default=None,
        help="在分类头前用 1×1 卷积压缩到指定通道数；默认关闭",
    )
    parser.add_argument(
        "--classifier-head",
        choices=SUPPORTED_CLASSIFIER_HEADS,
        default="flatten",
        help="分类头类型：flatten 为历史展平头，position 为位置感知共享头",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help=f"分类头 Dropout 概率，设为 0 可关闭，默认 {DEFAULT_DROPOUT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子，默认 {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="仅对训练集启用轻量旋转、平移、缩放和错切增强",
    )
    parser.add_argument(
        "--color-jitter",
        action="store_true",
        help="仅对训练集启用轻量亮度、对比度、饱和度和色相增强",
    )
    parser.add_argument(
        "--evaluate-clean-train",
        action="store_true",
        help="每轮额外在关闭增强和 Dropout 后评估训练集",
    )
    parser.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        help="验证 loss 停止改善时自动降低学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=(
            "AdamW 权重衰减系数；大于 0 时启用 AdamW，"
            f"默认 {DEFAULT_WEIGHT_DECAY}（使用 Adam 基线）"
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
        help=(f"交叉熵标签平滑系数，取值范围 [0, 1)，默认 {DEFAULT_LABEL_SMOOTHING}"),
    )
    args = parser.parse_args()

    train(
        config=TrainingConfig.from_namespace(args),
        model_path=args.model_path,
        num_conv_blocks=args.conv_blocks,
        bottleneck_channels=args.bottleneck_channels,
        classifier_head=args.classifier_head,
        dropout=args.dropout,
        seed=args.seed,
        augment=args.augment,
        color_jitter=args.color_jitter,
        evaluate_clean_train=args.evaluate_clean_train,
        reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        synthetic_train_dir=args.synthetic_train_dir,
        synthetic_ratio=args.synthetic_ratio,
        synthetic_validation_dir=args.synthetic_validation_dir,
    )
