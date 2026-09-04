import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, ToTensor

from datasets import CaptchaData, alphabet
from training_config import TrainingConfig
from training_curves import TrainingCurvePlotter

from .models import DenseNet

num_char = 4
num_class = len(alphabet)
model_path = "./stage1_densenet121/checkpoints/model-36.pth"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 第一阶段和第二阶段使用同一个 TrainingConfig 类，但可以有不同默认值。
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    epochs=30,
    batch_size=5,
    learning_rate=0.001,
)

os.makedirs("./stage1_densenet121/checkpoints", exist_ok=True)


def calculate_acc(output, target):
    output = output.view(-1, num_char, num_class).argmax(dim=2)
    target = target.view(-1, num_char, num_class).argmax(dim=2)
    return (output == target).all(dim=1).float().mean().item()


def calculate_loss(output, target, criterion):
    if isinstance(criterion, nn.CrossEntropyLoss):
        output = output.view(-1, num_class)
        target = target.view(-1, num_class).argmax(dim=1)
    return criterion(output, target)


def train(
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    classifier_only: bool = False,
    loss_name: str = "multi-label",
    backbone_learning_rate: float | None = None,
    pretrained: bool = True,
) -> None:
    if backbone_learning_rate is not None and backbone_learning_rate <= 0:
        raise ValueError("backbone_learning_rate 必须大于 0")

    torch.manual_seed(0)
    print(
        f"使用设备：{device} | 训练轮数：{config.epochs} | "
        f"批量大小：{config.batch_size} | 分类层学习率：{config.learning_rate} | "
        f"主干学习率：{backbone_learning_rate} | 损失函数：{loss_name} | "
        f"仅训练分类层：{classifier_only} | 使用预训练权重：{pretrained} | "
        f"显示曲线：{config.plot_curves}"
    )

    curve_plotter = TrainingCurvePlotter(
        enabled=config.plot_curves,
        title="DenseNet121 Training Curves",
    )

    transforms = Compose([ToTensor()])
    train_dataset = CaptchaData("./data/train", transform=transforms)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    test_dataset = CaptchaData("./data/test", transform=transforms)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False,
    )

    model = DenseNet(
        num_class=num_class,
        num_char=num_char,
        weights=models.DenseNet121_Weights.DEFAULT if pretrained else None,
    ).to(device)
    if classifier_only:
        model.densenet.features.requires_grad_(False)
        assert not any(
            parameter.requires_grad
            for parameter in model.densenet.features.parameters()
        )

    if classifier_only:
        parameters = model.densenet.classifier.parameters()
    elif backbone_learning_rate is not None:
        parameters = [
            {
                "params": model.densenet.features.parameters(),
                "lr": backbone_learning_rate,
            },
            {
                "params": model.densenet.classifier.parameters(),
                "lr": config.learning_rate,
            },
        ]
    else:
        parameters = model.parameters()

    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    expected_lrs = (
        [backbone_learning_rate, config.learning_rate]
        if backbone_learning_rate is not None and not classifier_only
        else [config.learning_rate]
    )
    assert [group["lr"] for group in optimizer.param_groups] == expected_lrs

    criterion = (
        nn.CrossEntropyLoss()
        if loss_name == "cross-entropy"
        else nn.MultiLabelSoftMarginLoss()
    )
    best_score = (-1.0, float("-inf"))

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_losses = []
        train_accuracies = []
        model.train()
        if classifier_only:
            model.densenet.features.eval()

        for img, target in train_data_loader:
            img = img.to(device)
            target = target.to(device)
            output = model(img)
            loss = calculate_loss(output, target, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_accuracies.append(calculate_acc(output, target))
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accuracies) / len(train_accuracies)

        test_loss_total = 0.0
        test_correct = 0.0
        test_size = 0
        model.eval()
        with torch.inference_mode():
            for img, target in test_data_loader:
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss = calculate_loss(output, target, criterion)
                current_batch_size = img.size(0)
                test_size += current_batch_size
                test_correct += calculate_acc(output, target) * current_batch_size
                test_loss_total += loss.item() * current_batch_size

        test_loss = test_loss_total / test_size
        test_acc = test_correct / test_size
        print(
            f"第 {epoch:02d}/{config.epochs} 轮 | "
            f"训练 loss: {train_loss:.4f} | "
            f"训练整图准确率: {train_acc:.2%} | "
            f"验证 loss: {test_loss:.4f} | "
            f"验证整图准确率: {test_acc:.2%} | "
            f"用时: {time.time() - start:.2f} 秒"
        )

        curve_plotter.update(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=test_loss,
            train_accuracy=train_acc,
            validation_accuracy=test_acc,
        )

        score = (round(test_acc, 6), -test_loss)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), model_path)
            assert os.path.getsize(model_path) > 0
            print(f"已保存当前最佳模型：{model_path}")

    curve_plotter.show()
    print(f"训练完成，最佳验证整图准确率：{best_score[0]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 DenseNet121 验证码模型")
    TrainingConfig.add_arguments(parser, DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="冻结 DenseNet 特征主干，仅训练分类层",
    )
    parser.add_argument(
        "--loss",
        choices=("multi-label", "cross-entropy"),
        default="multi-label",
        help="损失函数，默认 multi-label",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        help="DenseNet 特征主干的独立学习率",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不加载 ImageNet 预训练权重",
    )
    args = parser.parse_args()

    train(
        config=TrainingConfig.from_namespace(args),
        classifier_only=args.classifier_only,
        loss_name=args.loss,
        backbone_learning_rate=args.backbone_lr,
        pretrained=not args.no_pretrained,
    )
