"""第三阶段可变长度字符识别模型。"""

import torch
from torch import nn

from stage2_simple_cnn.models import SimpleCaptchaCNN

DEFAULT_CHARACTERS = "2345678abcdefgmnpwxy"
DEFAULT_BOTTLENECK_CHANNELS = 64
DEFAULT_LSTM_HIDDEN_SIZE = 64
SUPPORTED_SEQUENCE_MODELS = ("cnn", "bilstm")
SUPPORTED_SEQUENCE_WIDTHS = (22, 45)
DEFAULT_SEQUENCE_WIDTH = 22


class VariableLengthCaptchaCNN(nn.Module):
    """复用第二阶段 CNN，将横向特征图作为 CTC 时间序列。"""

    def __init__(
        self,
        characters: str = DEFAULT_CHARACTERS,
        sequence_model: str = "cnn",
        lstm_hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE,
        sequence_width: int = DEFAULT_SEQUENCE_WIDTH,
    ):
        super().__init__()
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("characters 必须包含不重复字符")
        if sequence_model not in SUPPORTED_SEQUENCE_MODELS:
            supported = ", ".join(SUPPORTED_SEQUENCE_MODELS)
            raise ValueError(f"sequence_model 必须是以下值之一：{supported}")
        if lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size 必须大于 0")
        if sequence_width not in SUPPORTED_SEQUENCE_WIDTHS:
            supported = ", ".join(str(value) for value in SUPPORTED_SEQUENCE_WIDTHS)
            raise ValueError(f"sequence_width 必须是以下值之一：{supported}")

        stage2_model = SimpleCaptchaCNN(
            num_class=len(characters),
            num_conv_blocks=4,
            bottleneck_channels=DEFAULT_BOTTLENECK_CHANNELS,
            classifier_head="position",
            dropout=0,
        )
        self.features = stage2_model.features
        if sequence_width == 45:
            pooling_indexes = [
                index
                for index, layer in enumerate(self.features)
                if isinstance(layer, nn.MaxPool2d)
            ]
            self.features[pooling_indexes[2]] = nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            )
        self.bottleneck = stage2_model.bottleneck

        self.characters = characters
        self.blank_index = len(characters)
        self.sequence_model = sequence_model
        self.lstm_hidden_size = lstm_hidden_size
        self.sequence_width = sequence_width
        if sequence_model == "bilstm":
            self.sequence_encoder: nn.Module = nn.LSTM(
                input_size=DEFAULT_BOTTLENECK_CHANNELS,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            classifier_input_size = lstm_hidden_size * 2
        else:
            self.sequence_encoder = nn.Identity()
            classifier_input_size = DEFAULT_BOTTLENECK_CHANNELS
        self.sequence_classifier = nn.Linear(
            classifier_input_size,
            len(characters) + 1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """把图片 `[B, 3, H, W]` 转换成序列 logits `[B, T, C]`。"""
        features = self.features(images)
        features = self.bottleneck(features)

        # 只汇聚高度，保留宽度作为从左到右的时间轴。
        sequence = features.mean(dim=2)
        sequence = sequence.transpose(1, 2)
        if isinstance(self.sequence_encoder, nn.LSTM):
            sequence, _state = self.sequence_encoder(sequence)
        else:
            sequence = self.sequence_encoder(sequence)
        return self.sequence_classifier(sequence)
