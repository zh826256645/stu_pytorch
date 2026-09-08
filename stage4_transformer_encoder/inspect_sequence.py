"""逐步观察 CNN 序列与一层显式实现的 Post-LN Transformer Encoder。

运行：uv run python -m stage4_transformer_encoder.inspect_sequence
"""

import torch
from torch.nn import functional as F

from stage3_variable_length.decoding import decode_ctc_path
from stage3_variable_length.training import compute_ctc_loss, encode_ctc_targets

from .models import CaptchaTransformer
from .position_encoding import add_position_encoding


def demo() -> None:
    # 复用第三阶段 CNN 结构，随机初始化；不加载或覆盖 checkpoint。
    model = CaptchaTransformer().eval()
    images = torch.randn(2, 3, 100, 180)
    with torch.no_grad():
        features = model.features(images)
        compressed = model.bottleneck(features)
        # [B, D, H, T] -> [B, D, T]：只平均高度，保留横向顺序。
        columns = compressed.mean(dim=2)
        # [B, D, T] -> [B, T, D]：每个横向位置对应一个 64 维向量。
        sequence = columns.transpose(1, 2)
        positioned = add_position_encoding(sequence)

    # 此教学检查从序列输入开始跟踪梯度，CNN 仍不参与反向传播。
    positioned.requires_grad_()
    # 三套独立参数，对同一输入做三种投影；每套参数在所有位置间共享。
    q_projection = model.encoder.query
    k_projection = model.encoder.key
    v_projection = model.encoder.value
    query = q_projection(positioned)
    key = k_projection(positioned)
    value = v_projection(positioned)

    # 每张图片独立计算：行是查询位置，列是被参考的位置。
    scores = query @ key.transpose(-2, -1) / (query.size(-1) ** 0.5)
    weights = scores.softmax(dim=-1)
    attended = weights @ value

    # 多头版本：拆分投影后的特征维度，所有头仍然看到全部 22 个位置。
    batch, length, dimension = query.shape
    num_heads = 4
    head_dim = dimension // num_heads
    q_heads = query.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    k_heads = key.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    v_heads = value.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    head_scores = q_heads @ k_heads.transpose(-2, -1) / (head_dim**0.5)
    head_weights = head_scores.softmax(dim=-1)
    head_outputs = head_weights @ v_heads
    # [B, 4, T, 16] -> [B, T, 4, 16] -> [B, T, 64]，不是对头求平均。
    merged = head_outputs.transpose(1, 2).reshape(batch, length, dimension)
    output_projection = model.encoder.output
    multihead_output = output_projection(merged)

    # Post-LN：先逐元素加回本子层输入，再对各位置的 64 维特征归一化。
    residual = positioned + multihead_output
    layer_norm = model.encoder.norm1
    normalized = layer_norm(residual)

    # FFN：所有位置共用参数，但分别变换自己的特征，不混合位置。
    ffn_expand = model.encoder.ffn[0]
    ffn_contract = model.encoder.ffn[2]
    ffn_hidden = torch.relu(ffn_expand(normalized))
    ffn_output = ffn_contract(ffn_hidden)
    ffn_residual = normalized + ffn_output
    ffn_norm = model.encoder.norm2
    encoded = ffn_norm(ffn_residual)

    # 每个时间步输出字符类别与 blank 的分数，不直接预测整串长度。
    classifier = model.classifier
    logits = classifier(encoded)
    # 教学分步计算必须与整理后的模型 forward 完全一致。
    with torch.no_grad():
        torch.testing.assert_close(encoded, model.encoder(positioned))
        torch.testing.assert_close(logits, model(images))
    # 仅为随机输入配两条合法示例标签，用于检查接口，不是真实训练样本。
    labels = ["a7", "aa2"]
    targets, target_lengths = encode_ctc_targets(labels, model.characters)
    loss = compute_ctc_loss(logits, targets, target_lengths, model.blank_index)

    assert features.shape == (2, 128, 3, 22)
    assert compressed.shape == (2, 64, 3, 22)
    assert columns.shape == (2, 64, 22)
    assert sequence.shape == (2, 22, 64)
    assert positioned.shape == sequence.shape
    for projection, projected in (
        (q_projection, query),
        (k_projection, key),
        (v_projection, value),
    ):
        assert projected.shape == (2, 22, 64)
        # 单独计算位置 0，结果与批量投影一致：此时尚未混合其他位置。
        torch.testing.assert_close(projected[:, 0], projection(positioned[:, 0]))
    assert q_projection.weight is not k_projection.weight
    assert k_projection.weight is not v_projection.weight
    assert q_projection.weight is not v_projection.weight
    assert scores.shape == weights.shape == (2, 22, 22)
    assert attended.shape == (2, 22, 64)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 22))
    # 用 PyTorch 内置注意力校验显式实现（默认无 mask、无 dropout）。
    torch.testing.assert_close(
        attended, F.scaled_dot_product_attention(query, key, value)
    )
    # 位置 0 的输出是本张图片所有 V 向量的加权和，不跨图片汇总。
    torch.testing.assert_close(
        attended[0, 0], (weights[0, 0, :, None] * value[0]).sum(dim=0)
    )
    assert q_heads.shape == k_heads.shape == v_heads.shape == (2, 4, 22, 16)
    assert head_scores.shape == head_weights.shape == (2, 4, 22, 22)
    assert multihead_output.shape == (2, 22, 64)
    torch.testing.assert_close(head_weights.sum(dim=-1), torch.ones(2, 4, 22))
    torch.testing.assert_close(
        head_outputs, F.scaled_dot_product_attention(q_heads, k_heads, v_heads)
    )
    # 检查分头/合头没有错置位置或特征顺序。
    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim
        torch.testing.assert_close(q_heads[:, head], query[:, :, start:end])
        torch.testing.assert_close(merged[:, :, start:end], head_outputs[:, head])
    assert residual.shape == normalized.shape == (2, 22, 64)
    torch.testing.assert_close(residual, positioned + multihead_output)
    # 独立按最后一维计算公式，验证不是跨位置或跨 batch 归一化。
    mean = residual.mean(dim=-1, keepdim=True)
    variance = residual.var(dim=-1, correction=0, keepdim=True)
    expected = (residual - mean) / torch.sqrt(variance + layer_norm.eps)
    expected = expected * layer_norm.weight + layer_norm.bias
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(normalized[0, 0], layer_norm(residual[0, 0]))
    assert ffn_hidden.shape == (2, 22, 128)
    assert ffn_output.shape == encoded.shape == (2, 22, 64)
    assert ffn_norm is not layer_norm
    # 单个位置独立跑 FFN，结果应与批量计算一致。
    torch.testing.assert_close(
        ffn_output[0, 0], ffn_contract(torch.relu(ffn_expand(normalized[0, 0])))
    )
    torch.testing.assert_close(encoded, ffn_norm(normalized + ffn_output))
    assert logits.shape == (2, 22, len(model.characters) + 1)
    assert target_lengths.tolist() == [2, 3]
    assert targets.tolist() == [model.characters.index(char) for char in "a7aa2"]
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    # CTC 先合并相邻重复，再去 blank；重复字符需要 blank 隔开。
    a_index = model.characters.index("a")
    assert (
        decode_ctc_path([a_index, a_index], model.characters, model.blank_index) == "a"
    )
    assert (
        decode_ctc_path(
            [a_index, model.blank_index, a_index], model.characters, model.blank_index
        )
        == "aa"
    )
    # 现在使用 CTC 梯度替代随机目标均方误差；仍不更新任何参数。
    loss.backward()
    assert positioned.grad is not None
    assert torch.isfinite(positioned.grad).all()
    for projection in (
        q_projection,
        k_projection,
        v_projection,
        output_projection,
        layer_norm,
        ffn_expand,
        ffn_contract,
        ffn_norm,
        classifier,
    ):
        for parameter in projection.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    # 用可区分的数值验证“第 t 个向量 = 第 t 列沿高度取平均”。
    probe = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    probe_sequence = probe.mean(dim=2).transpose(1, 2)
    torch.testing.assert_close(probe_sequence[0, 2], probe[0, :, :, 2].mean(dim=1))
    for name, tensor in (
        ("RGB 图片", images),
        ("四卷积块特征", features),
        ("压缩通道", compressed),
        ("平均高度", columns),
        ("横向视觉序列 [B, T, D]", sequence),
        ("加入位置编码 [B, T, D]", positioned),
        ("Q（查询向量）", query),
        ("K（键向量）", key),
        ("V（值向量）", value),
        ("位置匹配分数 [B, T, T]", scores),
        ("注意力权重 [B, T, T]", weights),
        ("加权汇总 [B, T, D]", attended),
        ("分头 Q/K/V [B, heads, T, head_dim]", q_heads),
        ("各头注意力权重 [B, heads, T, T]", head_weights),
        ("各头输出 [B, heads, T, head_dim]", head_outputs),
        ("合头 [B, T, D]", merged),
        ("输出投影 [B, T, D]", multihead_output),
        ("残差相加 [B, T, D]", residual),
        ("LayerNorm [B, T, D]", normalized),
        ("FFN 升维与 ReLU [B, T, 128]", ffn_hidden),
        ("FFN 降维 [B, T, D]", ffn_output),
        ("第二组残差 [B, T, D]", ffn_residual),
        ("第二个 LayerNorm / Encoder 输出 [B, T, D]", encoded),
        ("字符分数 logits [B, T, 类别数]", logits),
    ):
        print(f"{name}: {list(tensor.shape)}")
    print(f"示例标签：{labels} | CTC loss={loss.item():.4f}（随机输入，非训练指标）")
    print("Encoder、字符分类头、CTC 与梯度检查通过；未训练，CNN 仍不参与反向传播。")


if __name__ == "__main__":
    demo()
