import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """第一、二阶段共用的训练参数。

    每个阶段可以提供不同的默认值，但命令行参数名称和校验规则统一放在
    这个类中。这样新增公共训练参数时，只需要修改一个地方。
    """

    epochs: int
    batch_size: int
    learning_rate: float
    plot_curves: bool = False

    def __post_init__(self) -> None:
        """创建参数对象时立即检查数值是否合法。"""
        if self.epochs <= 0:
            raise ValueError("epochs 必须大于 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")

    @classmethod
    def add_arguments(
        cls,
        parser: argparse.ArgumentParser,
        defaults: "TrainingConfig",
    ) -> None:
        """给训练脚本添加两阶段共用的命令行参数。"""
        parser.add_argument(
            "--epochs",
            type=int,
            default=defaults.epochs,
            help=f"训练轮数，默认 {defaults.epochs}",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=defaults.batch_size,
            help=f"每批图片数量，默认 {defaults.batch_size}",
        )
        parser.add_argument(
            "--lr",
            dest="learning_rate",
            type=float,
            default=defaults.learning_rate,
            help=f"学习率，默认 {defaults.learning_rate}",
        )
        parser.add_argument(
            "--plot-curves",
            action="store_true",
            default=defaults.plot_curves,
            help="实时显示训练集和验证集的 loss、整图准确率曲线",
        )

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "TrainingConfig":
        """把 argparse 解析结果转换为经过校验的参数对象。"""
        return cls(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            plot_curves=args.plot_curves,
        )
