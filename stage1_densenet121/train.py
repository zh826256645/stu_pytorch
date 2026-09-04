import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, ToTensor

from datasets import CaptchaData, alphabet
from .models import DenseNet

batch_size = 5
base_lr = 0.001
max_epoch = 30
model_path = "./stage1_densenet121/checkpoints/model-36.pth"
num_char = 4
num_class = len(alphabet)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
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
    classifier_only=False,
    loss_name="multi-label",
    learning_rate=base_lr,
    backbone_learning_rate=None,
    pretrained=True,
    plot_curves=False,
):
    if learning_rate <= 0 or (
        backbone_learning_rate is not None and backbone_learning_rate <= 0
    ):
        raise ValueError("learning rates must be positive")

    torch.manual_seed(0)
    print(
        f"device: {device}|loss: {loss_name}|classifier_only: {classifier_only}"
        f"|pretrained: {pretrained}|plot_curves: {plot_curves}"
        f"|lr: {learning_rate}|backbone_lr: {backbone_learning_rate}"
    )

    plotter = None
    if plot_curves:
        import matplotlib.pyplot as plt

        plotter = plt
        plotter.ion()
        figure, (loss_axis, accuracy_axis) = plotter.subplots(
            1, 2, figsize=(12, 4)
        )
        figure.suptitle("Training Curves")
        loss_axis.set_title("Loss")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        train_loss_line, = loss_axis.plot([], [], label="train")
        test_loss_line, = loss_axis.plot([], [], label="test")
        loss_axis.legend()
        accuracy_axis.set_title("Exact-match Accuracy")
        accuracy_axis.set_xlabel("Epoch")
        accuracy_axis.set_ylabel("Accuracy")
        accuracy_axis.set_ylim(0, 1)
        train_accuracy_line, = accuracy_axis.plot([], [], label="train")
        test_accuracy_line, = accuracy_axis.plot([], [], label="test")
        accuracy_axis.legend()
        plotter.show(block=False)

    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    transforms = Compose([ToTensor()])
    train_dataset = CaptchaData("./data/train", transform=transforms)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    test_dataset = CaptchaData("./data/test", transform=transforms)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    model = DenseNet(
        num_class=num_class,
        num_char=num_char,
        weights=(
            models.DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        ),
    ).to(device)
    if classifier_only:
        model.densenet.features.requires_grad_(False)
        assert not any(
            parameter.requires_grad for parameter in model.densenet.features.parameters()
        )

    if classifier_only:
        parameters = model.densenet.classifier.parameters()
    elif backbone_learning_rate is not None:
        parameters = [
            {
                "params": model.densenet.features.parameters(),
                "lr": backbone_learning_rate,
            },
            {"params": model.densenet.classifier.parameters(), "lr": learning_rate},
        ]
    else:
        parameters = model.parameters()

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    expected_lrs = (
        [backbone_learning_rate, learning_rate]
        if backbone_learning_rate is not None and not classifier_only
        else [learning_rate]
    )
    assert [group["lr"] for group in optimizer.param_groups] == expected_lrs
    criterion = (
        nn.CrossEntropyLoss()
        if loss_name == "cross-entropy"
        else nn.MultiLabelSoftMarginLoss()
    )
    best_score = (-1.0, float("-inf"))

    for epoch in range(max_epoch):
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
        print(f"train_loss: {train_loss:.4}|train_acc: {train_acc:.4}")

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
        print(f"test_loss: {test_loss:.4}|test_acc: {test_acc:.4}")
        print(f"epoch: {epoch}|time: {time.time() - start:.4f}")

        if plotter is not None:
            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            epochs = history["epochs"]
            train_loss_line.set_data(epochs, history["train_loss"])
            test_loss_line.set_data(epochs, history["test_loss"])
            train_accuracy_line.set_data(epochs, history["train_acc"])
            test_accuracy_line.set_data(epochs, history["test_acc"])
            loss_axis.relim()
            loss_axis.autoscale_view()
            accuracy_axis.relim()
            accuracy_axis.set_ylim(0, 1)
            accuracy_axis.autoscale_view(scaley=False)
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plotter.pause(0.001)

        score = (round(test_acc, 6), -test_loss)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), model_path)
            assert os.path.getsize(model_path) > 0
            print(
                f"saved_best: epoch={epoch}|test_acc={test_acc:.4f}"
                f"|test_loss={test_loss:.4f}"
            )

    if plotter is not None:
        plotter.ioff()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="freeze DenseNet features and train only the classifier",
    )
    parser.add_argument(
        "--loss",
        choices=("multi-label", "cross-entropy"),
        default="multi-label",
    )
    parser.add_argument("--lr", type=float, default=base_lr)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="do not load ImageNet-pretrained DenseNet121 weights",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="show live training and validation curves",
    )
    args = parser.parse_args()
    train(
        args.classifier_only,
        args.loss,
        args.lr,
        args.backbone_lr,
        pretrained=not args.no_pretrained,
        plot_curves=args.plot_curves,
    )
