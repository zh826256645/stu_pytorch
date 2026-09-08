"""Pre/Post-LN 非训练对照：uv run python -m stage4_transformer_encoder.test_encoder。"""

import torch
from torch import nn

from .models import EncoderLayer


def demo():
    torch.manual_seed(0)
    sequence = torch.randn(2, 22, 64)
    for norm_first in (False, True):
        layer = EncoderLayer(norm_first=norm_first).eval()
        reference = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0,
            batch_first=True,
            norm_first=norm_first,
        ).eval()
        with torch.no_grad():
            reference.self_attn.in_proj_weight.copy_(
                torch.cat([layer.query.weight, layer.key.weight, layer.value.weight])
            )
            reference.self_attn.in_proj_bias.copy_(
                torch.cat([layer.query.bias, layer.key.bias, layer.value.bias])
            )
            reference.self_attn.out_proj.load_state_dict(layer.output.state_dict())
            reference.linear1.load_state_dict(layer.ffn[0].state_dict())
            reference.linear2.load_state_dict(layer.ffn[2].state_dict())
            reference.norm1.load_state_dict(layer.norm1.state_dict())
            reference.norm2.load_state_dict(layer.norm2.state_dict())
            torch.testing.assert_close(layer(sequence), reference(sequence))
            # 两条计算分支置零：Pre-LN 的层输出应完整保留残差输入。
            layer.output.weight.zero_()
            layer.output.bias.zero_()
            layer.ffn[2].weight.zero_()
            layer.ffn[2].bias.zero_()
            expected = sequence if norm_first else layer.norm2(layer.norm1(sequence))
            torch.testing.assert_close(layer(sequence), expected)
    print("Pre/Post-LN 与 PyTorch 单层实现一致，残差路径检查通过；未训练。")


if __name__ == "__main__":
    demo()
