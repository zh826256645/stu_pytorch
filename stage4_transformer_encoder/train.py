"""第四阶段正式训练：用户手动运行，测试集不参与，默认从随机初始化开始。"""

import argparse
import copy
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine, pil_to_tensor

from stage3_variable_length.data import ManifestCaptchaDataset
from stage3_variable_length.generate_captchas import render_variable_length_captcha
from stage3_variable_length.models import DEFAULT_CHARACTERS
from stage3_variable_length.train import (
    _format_per_length,
    _is_better_validation_result,
    build_lr_scheduler,
    choose_device,
    evaluate_model,
    run_training_epoch_with_diagnostics,
)

from .models import CaptchaTransformer

TRAIN_DIR = Path("data/stage3_variable_length_v2/train")
VALIDATION_DIR = Path("data/stage3_variable_length/validation")
MODEL_PATH = Path("stage4_transformer_encoder/checkpoints/model.pth")


class AugmentedDataset(Dataset):
    """只包装训练集；验证集和 BatchNorm 校准仍读取原图。"""

    def __init__(self, dataset, *, augment=True, rerender_probability=0.0, seed=0):
        if not 0 <= rerender_probability <= 1:
            raise ValueError("rerender-probability 必须在 [0, 1] 内")
        self.dataset = dataset
        self.augment = augment
        self.rerender_probability = rerender_probability
        # ponytail: 当前 DataLoader 为单进程；增加 workers 时需分别初始化随机流。
        self.rng = random.Random(seed)
        self.fonts = []
        if rerender_probability:
            self.fonts = sorted({Path(r["font"]) for r in dataset.records if r.get("font")})
            if not self.fonts:
                raise ValueError("在线重渲染需要训练记录中的字体路径")
            for font in self.fonts:
                if not font.is_file():
                    raise ValueError(f"在线重渲染字体不存在：{font}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.rerender_probability and self.rng.random() < self.rerender_probability:
            label = str(self.dataset.records[index]["label"])
            rendered = render_variable_length_captcha(
                label,
                self.rng.choice(self.fonts),
                random.Random(self.rng.getrandbits(64)),
                width=self.dataset.width,
                height=self.dataset.height,
            )
            image = pil_to_tensor(rendered).to(dtype=torch.float32).div(255)
        else:
            image, label = self.dataset[index]
        if not self.augment:
            return image, label
        height, width = image.shape[-2:]
        return affine(
            image,
            float(torch.empty(()).uniform_(-3, 3)),
            [int(torch.empty(()).uniform_(-0.03 * width, 0.03 * width)),
             int(torch.empty(()).uniform_(-0.03 * height, 0.03 * height))],
            float(torch.empty(()).uniform_(0.95, 1.05)),
            [float(torch.empty(()).uniform_(-3, 3)), 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
        ), label


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--sequence-width", type=int, choices=(22, 45), default=22)
    parser.add_argument("--length-6-loss-weight", type=float, default=2.0)
    parser.add_argument("--new-renders-only", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--rerender-probability", type=float, default=0.0,
        help="训练时用同标签的新渲染替换原图的概率，默认关闭",
    )
    parser.add_argument(
        "--pre-norm",
        dest="norm_first",
        action="store_true",
        help="使用 Pre-LN；默认 Post-LN",
    )
    parser.add_argument(
        "--no-position-encoding",
        dest="use_position_encoding",
        action="store_false",
        help="关闭正弦位置编码（默认开启），用于消融对照",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="auto"
    )
    parser.add_argument("--threads", type=int, default=4)
    parsed = parser.parse_args(args)
    if min(parsed.epochs, parsed.batch_size, parsed.threads) <= 0:
        parser.error("epochs、batch-size、threads 必须为正数")
    if not math.isfinite(parsed.lr) or parsed.lr <= 0:
        parser.error("lr 必须为有限正数")
    if not 0 <= parsed.dropout < 1:
        parser.error("dropout 必须满足 0 <= dropout < 1")
    if not 0 <= parsed.rerender_probability <= 1:
        parser.error("rerender-probability 必须在 [0, 1] 内")
    if parsed.ffn_dim <= 0:
        parser.error("ffn-dim 必须大于 0")
    if (
        not math.isfinite(parsed.length_6_loss_weight)
        or parsed.length_6_loss_weight <= 0
    ):
        parser.error("length-6-loss-weight 必须为有限正数")
    return parsed


def recalibrate_batchnorm(model, loader, device):
    """只在副本上用训练集重算 BN 统计，保持优化器所引用的模型不变。"""
    calibrated = copy.deepcopy(model).eval()
    for module in calibrated.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
            module.train()
    count = 0
    with torch.inference_mode():
        for images, _ in loader:
            calibrated(images.to(device))
            count += 1
    if count == 0:
        raise ValueError("BatchNorm 校准训练集不能为空")
    return calibrated.eval()


def save_checkpoint(model, path, args, epoch, loss, result):
    """原子替换本阶段唯一最佳权重；不是可恢复优化器状态的训练快照。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "version": 1,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "characters": model.characters,
            "blank_index": model.blank_index,
            "model_config": {
                "sequence_model": "transformer",
                "sequence_width": model.sequence_width,
                "d_model": 64,
                "num_heads": 4,
                "dim_feedforward": 128,
                "ffn_dim": model.encoder.ffn_dim,
                "num_layers": 1,
                "norm_first": model.encoder.norm_first,
                "dropout": model.encoder.dropout.p,
                "use_position_encoding": model.use_position_encoding,
            },
            "training_config": {
                **vars(args),
                "weight_decay": 0.0,
                "length_6_loss_weight": args.length_6_loss_weight,
                "recalibrate_batchnorm": True,
                "reduce_lr_on_plateau": True,
                "train_dir": str(TRAIN_DIR),
                "validation_dir": str(VALIDATION_DIR),
            },
            "epoch": epoch,
            "validation_loss": loss,
            "validation_exact_match_accuracy": result.exact_match_accuracy,
            "validation_character_error_rate": result.character_error_rate,
            "validation_per_length_accuracy": {
                length: metrics.accuracy
                for length, metrics in result.per_length.items()
            },
        },
        temporary,
    )
    temporary.replace(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    train_data = ManifestCaptchaDataset(TRAIN_DIR)
    validation_data = ManifestCaptchaDataset(VALIDATION_DIR)
    if args.new_renders_only:
        train_data.records = [
            r for r in train_data.records if r.get("source") == "new_render"
        ]
        train_data.labels = [str(r["label"]) for r in train_data.records]
        if not train_data.records:
            raise ValueError("没有找到 source=new_render 的训练图片")
    for dataset, split in ((train_data, "train"), (validation_data, "validation")):
        if dataset.split != split or len(dataset) == 0:
            raise ValueError(f"需要非空的 {split} 数据集")
        if dataset.characters != DEFAULT_CHARACTERS or (
            dataset.width,
            dataset.height,
        ) != (180, 100):
            raise ValueError("需要原有字符表和 180×100 图片")
        if any(
            not 2 <= len(str(r["label"])) <= 6
            or not set(str(r["label"])) <= set(DEFAULT_CHARACTERS)
            for r in dataset.records
        ):
            raise ValueError("标签必须为字符表内的 2～6 位字符串")
    training_dataset = (
        AugmentedDataset(
            train_data, augment=args.augment,
            rerender_probability=args.rerender_probability, seed=args.seed,
        )
        if args.augment or args.rerender_probability else train_data
    )
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    clean_loader = DataLoader(train_data, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    model = CaptchaTransformer(
        dropout=args.dropout,
        use_position_encoding=args.use_position_encoding,
        norm_first=args.norm_first,
        ffn_dim=args.ffn_dim,
        sequence_width=args.sequence_width,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = build_lr_scheduler(optimizer)
    best_accuracy, best_cer, best_loss, best_epoch = -1.0, float("inf"), float("inf"), 0
    print(
        f"设备={device} | 参数量={sum(p.numel() for p in model.parameters())} | "
        f"训练={len(train_data)} | 验证={len(validation_data)}",
        flush=True,
    )
    print(f"{vars(args)} | BN 重校准 | 自动学习率 | greedy", flush=True)
    print(f"随机初始化；最佳权重将覆盖 {MODEL_PATH}；不读取测试集。", flush=True)
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        diagnostics = run_training_epoch_with_diagnostics(
            model,
            train_loader,
            optimizer,
            device,
            length_6_loss_weight=args.length_6_loss_weight,
        )
        if diagnostics.nonfinite_loss_count or diagnostics.nonfinite_gradient_count:
            raise RuntimeError("发现非有限 loss/梯度，停止且不覆盖最佳权重")
        calibrated = recalibrate_batchnorm(model, clean_loader, device)
        train_loss, train_result, _ = evaluate_model(calibrated, clean_loader, device)
        val_loss, result, _ = evaluate_model(calibrated, validation_loader, device)
        if not all(math.isfinite(v) for v in (train_loss, val_loss)):
            raise RuntimeError("评价 loss 非有限值，不覆盖最佳权重")
        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr:.6f} | "
            f"加权训练 loss={diagnostics.average_loss:.4f} | 验证 loss={val_loss:.4f}",
            flush=True,
        )
        print(
            f"  训练准确率={train_result.exact_match_accuracy:.2%} | "
            f"CER={train_result.character_error_rate:.2%} | 无权重 loss={train_loss:.4f}",
            flush=True,
        )
        print(
            f"  验证准确率={result.exact_match_accuracy:.2%} | CER={result.character_error_rate:.2%} | "
            f"准确率差距={train_result.exact_match_accuracy - result.exact_match_accuracy:.2%}",
            flush=True,
        )
        print(f"  验证按长度：{_format_per_length(result)}", flush=True)
        if _is_better_validation_result(
            result,
            val_loss,
            best_accuracy=best_accuracy,
            best_character_error_rate=best_cer,
            best_loss=best_loss,
        ):
            save_checkpoint(calibrated, MODEL_PATH, args, epoch, val_loss, result)
            best_accuracy, best_cer, best_loss, best_epoch = (
                result.exact_match_accuracy,
                result.character_error_rate,
                val_loss,
                epoch,
            )
            print("  已更新最佳验证权重", flush=True)
        scheduler.step(val_loss)
        if optimizer.param_groups[0]["lr"] < lr:
            print(f"  下一轮 lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)
    print(
        f"完成：最佳轮次={best_epoch} | 验证准确率={best_accuracy:.2%} | "
        f"CER={best_cer:.2%} | loss={best_loss:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
