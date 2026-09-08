"""一层显式 Post-LN Encoder：64 维、4 头、128 维 FFN，可选残差分支 Dropout。"""

import torch
from torch import nn

from stage3_variable_length.models import DEFAULT_CHARACTERS, VariableLengthCaptchaCNN

from .position_encoding import add_position_encoding


class EncoderLayer(nn.Module):
    def __init__(
        self, dropout: float = 0.0, norm_first: bool = False, ffn_dim: int = 128
    ):
        super().__init__()
        if not isinstance(norm_first, bool):
            raise ValueError("norm_first 必须是布尔值")
        self.norm_first = norm_first
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须满足 0 <= dropout < 1")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim 必须大于 0")
        self.ffn_dim = ffn_dim
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.output = nn.Linear(64, 64)
        self.norm1 = nn.LayerNorm(64)
        self.ffn = nn.Sequential(
            nn.Linear(64, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, 64)
        )
        self.norm2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, length, _ = sequence.shape
        attention_input = self.norm1(sequence) if self.norm_first else sequence
        query = (
            self.query(attention_input).reshape(batch, length, 4, 16).transpose(1, 2)
        )
        key = self.key(attention_input).reshape(batch, length, 4, 16).transpose(1, 2)
        value = (
            self.value(attention_input).reshape(batch, length, 4, 16).transpose(1, 2)
        )
        weights = (query @ key.transpose(-2, -1) / (16**0.5)).softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, 64)
        residual = sequence + self.dropout(self.output(attended))
        if self.norm_first:
            return residual + self.dropout(self.ffn(self.norm2(residual)))
        normalized = self.norm1(residual)
        return self.norm2(normalized + self.dropout(self.ffn(normalized)))


class CaptchaTransformer(nn.Module):
    def __init__(
        self,
        characters: str = DEFAULT_CHARACTERS,
        dropout: float = 0.0,
        use_position_encoding: bool = True,
        norm_first: bool = False,
        ffn_dim: int = 128,
        sequence_width: int = 22,
    ):
        super().__init__()
        if not isinstance(use_position_encoding, bool):
            raise ValueError("use_position_encoding 必须是布尔值")
        self.use_position_encoding = use_position_encoding
        baseline = VariableLengthCaptchaCNN(
            characters=characters, sequence_width=sequence_width
        )
        self.sequence_width = baseline.sequence_width
        self.features = baseline.features
        self.bottleneck = baseline.bottleneck
        self.classifier = baseline.sequence_classifier
        self.encoder = EncoderLayer(dropout, norm_first, ffn_dim)
        self.characters = characters
        self.blank_index = len(characters)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.bottleneck(self.features(images))
        sequence = features.mean(dim=2).transpose(1, 2)
        if self.use_position_encoding:
            sequence = add_position_encoding(sequence)
        return self.classifier(self.encoder(sequence))
