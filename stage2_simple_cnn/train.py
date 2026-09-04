import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomAffine, ToTensor

from datasets import CaptchaData, alphabet
from training_config import TrainingConfig
from training_curves import TrainingCurvePlotter
from training_metrics import calculate_accuracy_counts

from .models import SimpleCaptchaCNN

# 当前验证码固定为 4 个字符。
NUM_CHAR = 4
# 字符集为 0-9 和 a-z，共 36 类。
NUM_CLASS = len(alphabet)
DEFAULT_MODEL_PATH = Path("stage2_simple_cnn/checkpoints/model.pth")

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


def get_device() -> torch.device:
    """按照 CUDA、Apple MPS、CPU 的顺序选择训练设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_image_transform(*, augment: bool) -> Compose:
    """构造图片预处理；训练时可选择轻量仿射增强。"""
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
    transforms.append(ToTensor())
    return Compose(transforms)


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
    augment: bool = False,
    reduce_lr_on_plateau: bool = False,
) -> None:
    """训练简单 CNN，并保存验证集表现最好的模型。"""
    # TrainingConfig 创建时已经统一检查 epochs、batch_size 和学习率。
    torch.manual_seed(0)
    device = get_device()
    print(
        f"使用设备：{device} | 训练轮数：{config.epochs} | "
        f"批量大小：{config.batch_size} | 初始学习率：{config.learning_rate} | "
        f"学习率调度：{reduce_lr_on_plateau} | 数据增强：{augment} | "
        f"显示曲线：{config.plot_curves}"
    )

    # plot_curves=False 时，这个共用模块不会导入或打开 Matplotlib。
    curve_plotter = TrainingCurvePlotter(
        enabled=config.plot_curves,
        title="Simple CNN Training Curves",
    )

    # 只增强训练集，验证集始终只执行 ToTensor，确保不同实验的指标可比较。
    train_transform = build_image_transform(augment=augment)
    validation_transform = build_image_transform(augment=False)
    train_dataset = CaptchaData("./data/train", transform=train_transform)
    test_dataset = CaptchaData("./data/test", transform=validation_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SimpleCaptchaCNN(
        num_class=NUM_CLASS,
        num_char=NUM_CHAR,
    ).to(device)

    # 每个字符位置都是一个 36 分类问题，因此使用交叉熵损失。
    criterion = nn.CrossEntropyLoss()
    # Adam 会根据计算出的梯度更新卷积层和全连接层的参数。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    scheduler = build_lr_scheduler(optimizer) if reduce_lr_on_plateau else None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, config.epochs + 1):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy, train_character_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        test_loss, test_accuracy, test_character_accuracy = run_epoch(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"第 {epoch:02d}/{config.epochs} 轮 | "
            f"学习率: {current_learning_rate:.6f} | "
            f"训练 loss: {train_loss:.4f} | "
            f"训练整图准确率: {train_accuracy:.2%} | "
            f"训练单字符准确率: {train_character_accuracy:.2%} | "
            f"验证 loss: {test_loss:.4f} | "
            f"验证整图准确率: {test_accuracy:.2%} | "
            f"验证单字符准确率: {test_character_accuracy:.2%}"
        )

        # 两个阶段通过同一个绘图模块显示相同含义的指标。
        curve_plotter.update(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=test_loss,
            train_accuracy=train_accuracy,
            validation_accuracy=test_accuracy,
            train_character_accuracy=train_character_accuracy,
            validation_character_accuracy=test_character_accuracy,
        )

        if scheduler is not None:
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step(test_loss)
            next_learning_rate = optimizer.param_groups[0]["lr"]
            if next_learning_rate < previous_learning_rate:
                print(
                    "验证 loss 停止改善，学习率调整："
                    f"{previous_learning_rate:.6f} -> {next_learning_rate:.6f}"
                )

        # 只保存验证集整图准确率最高的权重。
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"已保存当前最佳模型：{model_path}")

    # 启用 --plot-curves 时，训练结束后等待用户关闭曲线窗口。
    curve_plotter.show()
    print(f"训练完成，最佳验证整图准确率：{best_accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练第一个简单验证码 CNN")
    # --epochs、--batch-size、--lr、--plot-curves 由共用参数类统一添加。
    TrainingConfig.add_arguments(parser, DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="最佳模型权重保存路径",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="仅对训练集启用轻量旋转、平移、缩放和错切增强",
    )
    parser.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        help="验证 loss 停止改善时自动降低学习率",
    )
    args = parser.parse_args()

    train(
        config=TrainingConfig.from_namespace(args),
        model_path=args.model_path,
        augment=args.augment,
        reduce_lr_on_plateau=args.reduce_lr_on_plateau,
    )
