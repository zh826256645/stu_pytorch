import torch
from torch import nn


class SimpleCaptchaCNN(nn.Module):
    """用于识别固定 4 位验证码的入门 CNN。

    输入形状：
        [批量大小, 3, 100, 180]

    输出形状：
        [批量大小, 4, 36]

    输出中的 4 表示验证码的 4 个字符位置，36 表示每个位置有
    0-9、a-z 共 36 个候选类别。
    """

    def __init__(self, num_class: int = 36, num_char: int = 4):
        super().__init__()
        self.num_class = num_class
        self.num_char = num_char

        # features 负责从原始 RGB 图片中提取图像特征。
        self.features = nn.Sequential(
            # 输入：[B, 3, 100, 180]
            # 3 是 RGB 三个颜色通道，16 是这一层输出的特征图数量。
            # padding=1 配合 3x3 卷积核，使图片的高和宽保持不变。
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
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
            nn.ReLU(),
            # 再次把高和宽缩小一半：
            # [B, 32, 50, 90] -> [B, 32, 25, 45]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 控制变量实验：用一次较温和的平均池化压缩送入全连接层的
            # 空间尺寸，卷积层、激活函数和其他训练配置保持不变。
            # PyTorch 默认向下取整，因此 25x45 会变为 12x22。
            # [B, 32, 25, 45] -> [B, 32, 12, 22]
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 平均池化后，每张图片具有 32 * 12 * 22 个特征。
        # 控制变量实验：只在全连接层前加入 Dropout(0.1)，训练时随机
        # 丢弃 10% 的输入特征，其他模型结构和训练配置保持不变。
        # 这里不添加 Softmax，因为 CrossEntropyLoss 内部会完成相应计算。
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=32 * 12 * 22,
                out_features=num_char * num_class,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义数据从输入到输出的前向传播过程。"""
        # 卷积和平均池化后的形状为 [B, 32, 12, 22]。
        x = self.features(x)

        # 从第 1 维开始展平，保留第 0 维的批量大小：
        # [B, 32, 12, 22] -> [B, 32 * 12 * 22]
        x = torch.flatten(x, start_dim=1)

        # 得到每张验证码的 4 * 36 个原始分类分数（logits）。
        x = self.classifier(x)

        # 将扁平输出整理成“4 个字符位置，每个位置 36 类”：
        # [B, 144] -> [B, 4, 36]
        return x.view(x.size(0), self.num_char, self.num_class)
