import torch
from torch import nn

SUPPORTED_CONV_BLOCKS = (3, 4)
SUPPORTED_CLASSIFIER_HEADS = ("flatten", "position")
DEFAULT_DROPOUT = 0.1


class HorizontalPositionPool(nn.Module):
    """将特征图等宽分成若干字符区域，并分别汇聚为通道向量。"""

    def __init__(self, num_positions: int):
        super().__init__()
        if num_positions <= 0:
            raise ValueError("num_positions 必须大于 0")
        self.num_positions = num_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把 [B, C, H, W] 汇聚成 [B, num_positions, C]。"""
        width = x.size(3)
        if width < self.num_positions:
            raise ValueError("特征图宽度不能小于字符位置数量")

        pooled_positions = []
        for position in range(self.num_positions):
            start = position * width // self.num_positions
            end = (position + 1) * width // self.num_positions
            pooled_positions.append(x[:, :, :, start:end].mean(dim=(2, 3)))
        return torch.stack(pooled_positions, dim=1)


class SimpleCaptchaCNN(nn.Module):
    """用于识别固定 4 位验证码的入门 CNN。

    输入形状：
        [批量大小, 3, 100, 180]

    输出形状：
        [批量大小, 4, 36]

    输出中的 4 表示验证码的 4 个字符位置，36 表示每个位置有
    0-9、a-z 共 36 个候选类别。

    默认使用已经验证过的三个卷积块；实验时可通过 ``num_conv_blocks=4``
    增加第四个卷积块。还可通过 ``bottleneck_channels`` 在分类头前增加
    1×1 卷积瓶颈，并通过 ``classifier_head`` 比较传统展平分类头和
    按四个横向字符位置共享参数的位置感知分类头。
    """

    def __init__(
        self,
        num_class: int = 36,
        num_char: int = 4,
        num_conv_blocks: int = 3,
        bottleneck_channels: int | None = None,
        classifier_head: str = "flatten",
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须大于等于 0 且小于 1")
        if num_conv_blocks not in SUPPORTED_CONV_BLOCKS:
            supported = ", ".join(str(value) for value in SUPPORTED_CONV_BLOCKS)
            raise ValueError(f"num_conv_blocks 必须是以下值之一：{supported}")
        if classifier_head not in SUPPORTED_CLASSIFIER_HEADS:
            supported = ", ".join(SUPPORTED_CLASSIFIER_HEADS)
            raise ValueError(f"classifier_head 必须是以下值之一：{supported}")

        self.num_class = num_class
        self.num_char = num_char
        self.num_conv_blocks = num_conv_blocks
        self.classifier_head = classifier_head
        self.dropout = dropout

        # features 负责从原始 RGB 图片中提取图像特征。
        feature_layers: list[nn.Module] = [
            # 输入：[B, 3, 100, 180]
            # 3 是 RGB 三个颜色通道，16 是这一层输出的特征图数量。
            # padding=1 配合 3x3 卷积核，使图片的高和宽保持不变。
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            # 归一化每个卷积通道的激活值，使训练过程更加稳定。
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 池化会把高和宽都缩小一半：
            # [B, 16, 100, 180] -> [B, 16, 50, 90]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二层卷积把通道数从 16 增加到 32：
            # [B, 16, 50, 90] -> [B, 32, 50, 90]
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 再次把高和宽缩小一半：
            # [B, 32, 50, 90] -> [B, 32, 25, 45]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三个卷积块进一步提取字符笔画、边缘组合和干扰线特征，
            # 同时把通道数从 32 增加到 64。
            # [B, 32, 25, 45] -> [B, 64, 25, 45]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [B, 64, 25, 45] -> [B, 64, 12, 22]
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        if num_conv_blocks == 4:
            feature_layers.extend(
                [
                    # 第四个卷积块把通道数从 64 增加到 128，并继续组合
                    # 高层字符特征。只池化高度，保留横向字符位置信息：
                    # [B, 64, 12, 22] -> [B, 128, 12, 22]
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    # [B, 128, 12, 22] -> [B, 128, 6, 22]
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                    # [B, 128, 6, 22] -> [B, 128, 3, 22]
                    nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
                ]
            )
            classifier_channels = 128
            classifier_height = 3
        else:
            # 三卷积块基线只继续压缩高度，不压缩宽度。
            # [B, 64, 12, 22] -> [B, 64, 6, 22]
            feature_layers.append(nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)))
            classifier_channels = 64
            classifier_height = 6

        self.features = nn.Sequential(*feature_layers)

        if bottleneck_channels is None:
            # 默认不改变现有基线结构，确保旧权重仍可按三卷积块模型加载。
            self.bottleneck = nn.Identity()
            classifier_input_channels = classifier_channels
        else:
            if not 1 <= bottleneck_channels <= classifier_channels:
                raise ValueError(
                    "bottleneck_channels 必须大于等于 1，且不能超过卷积特征通道数 "
                    f"{classifier_channels}"
                )
            # 1×1 卷积只压缩通道，不改变 6×22 或 3×22 的空间布局。
            self.bottleneck = nn.Sequential(
                nn.Conv2d(
                    in_channels=classifier_channels,
                    out_channels=bottleneck_channels,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
            )
            classifier_input_channels = bottleneck_channels

        self.bottleneck_channels = bottleneck_channels
        self.feature_shape = (classifier_input_channels, classifier_height, 22)

        # flatten 保留历史分类头，确保默认模型和旧权重完全兼容。
        # position 将宽度方向汇聚成 4 个字符位置，并让四个位置共享同一个
        # 字符分类器，减少参数并加入从左到右的任务先验。
        if classifier_head == "flatten":
            flattened_features = classifier_input_channels * classifier_height * 22
            self.position_pool = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(
                    in_features=flattened_features,
                    out_features=num_char * num_class,
                ),
            )
        else:
            self.position_pool = HorizontalPositionPool(num_char)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(
                    in_features=classifier_input_channels,
                    out_features=num_class,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义数据从输入到输出的前向传播过程。"""
        x = self.features(x)
        x = self.bottleneck(x)

        if self.classifier_head == "position":
            # 把宽度等分成 4 个字符区域并汇聚：[B, C, H, 22] -> [B, 4, C]。
            # 自定义分段均值避免 MPS 不支持 22 -> 4 非整除 AdaptiveAvgPool 的限制。
            x = self.position_pool(x)
            # Linear 作用于最后一维，四个位置共享同一组 C -> 36 参数。
            return self.classifier(x)

        # 历史展平分类头：[B, C, H, 22] -> [B, C * H * 22]。
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_char, self.num_class)
