# 第四阶段：CNN + Transformer Encoder + CTC

[返回项目首页](../../README.md) · [环境准备](../../README.md#环境准备)

本文所有命令和路径均以项目根目录为基准。

本阶段沿用第三阶段的数据划分、22 个横向时间步和评估口径，保留 CNN 与 CTC，将 BiLSTM
替换为 Transformer Encoder。目标是跑通并理解自注意力序列识别；验证整串准确率、CER 和
参数量用于观察比较，不要求超过 BiLSTM 的 94.00%。测试集继续封存。

由 AI 逐步实现，再结合代码引导学习：CNN 横向视觉序列 → 位置编码 → Q/K/V 投影、分头、
注意力计算与合头 → 残差连接与 LayerNorm → 前馈网络（FFN）→ 字符分数与 CTC。
注意力使用显式的最小教学实现，线性层与 LayerNorm 使用 PyTorch 组件，不扩展为通用框架。
先检查前向传播，再接 CTC 与小样本过拟合，最后进行正式实验。

## 第一步：图片变成序列

运行最小形状检查（随机输入与随机初始化，不读取数据集或 checkpoint）：

```bash
uv run python -m stage4_transformer_encoder.inspect_sequence
```

复用第三阶段 CNN，数据形状依次为：

```text
[B, 3, 100, 180] → [B, 128, 3, 22] → [B, 64, 3, 22]
                 → [B, 64, 22] → [B, 22, 64]
```

`mean(dim=2)` 只汇聚高度；`transpose(1, 2)` 把通道移到最后一维。
输出中的 22 是从左到右的视觉位置数，不是字符数；64 是每个位置的特征维度。
本步骤尚未加入位置编码、Encoder 或训练，不代表第四阶段模型已经完成。

## 第二步：位置编码

`stage4_transformer_encoder/position_encoding.py` 使用固定正弦/余弦位置编码，不引入可训练参数。
对横向位置 `p` 和特征维度对 `2i, 2i+1`：

```text
PE[p, 2i]   = sin(p / 10000^(2i/D))
PE[p, 2i+1] = cos(p / 10000^(2i/D))
输出 = CNN 横向视觉序列 + PE
```

每个位置获得一个 D=64 维向量，不同维度对使用不同频率。同一位置在不同图片中使用相同编码。
编码通过逐元素相加融入视觉特征，不是拼接，也不是新增字符类别；输出仍为 `[B, 22, 64]`。
这是位置线索，不是位置间的关联计算；后者将在自注意力中学习。

运行 `uv run python -m stage4_transformer_encoder.position_encoding` 检查位置值、形状与梯度。
第一步的 `inspect_sequence` 命令现在也显示加入位置编码后的形状；仍未训练或读取 checkpoint。

## 第三步：Q/K/V 投影

`inspect_sequence` 现在继续对加入位置编码的序列应用三套独立的 `nn.Linear(64, 64)`：

```python
query = q_projection(positioned)
key = k_projection(positioned)
value = v_projection(positioned)
```

三者均为 `[B, 22, 64]`。Q（Query）与 K（Key）将用于计算位置之间的匹配分数，
V（Value）是之后按注意力权重汇总的信息。它们来自同一序列，因此这里准备的是自注意力。
每套投影参数在所有位置间共享，但 Q/K/V 三套参数相互独立；投影仅作用于最后一个维度，
本步骤尚未混合不同位置的信息。与固定位置编码不同，这些投影参数需要在后续训练中学习。

仍运行 `uv run python -m stage4_transformer_encoder.inspect_sequence`，检查形状、
逐位置投影一致性和三套投影的梯度。此处使用的平方均值仅用于反向传播检查，不是识别任务的
训练目标；未更新参数，未计算注意力权重，也未进行多头拆分。

## 第四步：单头自注意力

在 Q/K/V 后加入三步计算：

```python
scores = query @ key.transpose(-2, -1) / (query.size(-1) ** 0.5)
weights = scores.softmax(dim=-1)
attended = weights @ value
```

`scores` 和 `weights` 均为 `[B, 22, 22]`：每行对应一个查询位置，每列对应一个被参考位置。
点积衡量 Q/K 的匹配程度；除以 `sqrt(64)=8` 用于缓解维度增加造成的分数尺度增大与 softmax
过度饱和。softmax 沿最后一维把每行分数转为非负、总和为 1 的权重。
输出为 `[B, 22, 64]`，每个位置用自己的一行权重汇总同一张图片的全部 V 向量。
这是首次显式混合不同序列位置的信息；CNN 本身此前已通过卷积感受野聚合空间邻域。

当前固定尺寸输入不需要补齐 mask，双向 Encoder 也不使用遮挡右侧位置的因果 mask。
权重来自随机初始化模型，尚无学到的识别意义。`inspect_sequence` 现检查权重行和、
加权汇总、与 PyTorch 内置注意力的一致性，以及注意力输出到 Q/K/V 投影的反向传播；
平方均值仍仅用于梯度检查，未更新参数，尚未加入多头、输出投影、残差或 LayerNorm。

## 第五步：多头注意力

在保留单头示例的同时，`inspect_sequence` 添加 4 头教学版本。Q/K/V 各自从 `[B, 22, 64]`
变为 `[B, 4, 22, 16]`：拆的是投影后的特征维度，不是序列长度，每个头仍可参考全部 22 个位置。
Q/K/V 投影的每个输出维度都可以使用完整的 64 维输入，因此这不是把原始输入硬分给不同头。

各头独立计算注意力，分数除以每头维度的平方根 `sqrt(16)=4`，权重形状为 `[B, 4, 22, 22]`。
将各头的 16 维输出拼回 64 维，再经可训练的 `Linear(64, 64)` 输出投影混合各头信息。
多头不是复制四份相同结果，也不是对四个头求平均；各头可能学到不同关联，但不预先指定其职责。

运行同一命令检查每个头的权重行和、内置注意力对照、分头/合头顺序，以及输出投影到 Q/K/V 的
梯度。仍未训练；下一步加入残差连接与 LayerNorm。

## 第六步：残差连接与 LayerNorm

当前教学版本采用 Post-LN（先残差相加，再归一化），沿着已完成的数据流继续添加：

```python
residual = positioned + multihead_output
layer_norm = nn.LayerNorm(64)
normalized = layer_norm(residual)
```

残差把注意力子层输入直接加回输出，让子层可以学习对输入的修正，并提供直接的梯度路径。
这是逐元素相加而非拼接，两侧都为 `[B, 22, 64]`。注意力输出为零时，残差相加结果等于输入，
但随后的 LayerNorm 仍会变换它；不能把整个 Post-LN 子层说成恒等映射。

`LayerNorm(64)` 对每张图片、每个位置各自的 64 个特征计算均值与方差，先标准化，
再应用可训练的逐维缩放与偏移。它不跨图片或位置统计，不使用 BatchNorm 的运行均值/方差；
训练和推理使用相同的归一化计算。由于 epsilon 及可训练的缩放/偏移，最终结果不保证始终
严格满足均值 0、方差 1。输出形状保持不变。

`inspect_sequence` 检查残差结果、LayerNorm 手算公式及单位置独立计算的一致性，
并通过随机目标的均方误差验证梯度能传回 LayerNorm、输出投影、Q/K/V 和序列输入。
该误差仅用于检查，CNN 不参与反向传播，未更新参数；下一步再加入 FFN，形成完整 Encoder 层。

## 第七步：FFN 与完整 Encoder 层

FFN（前馈网络）接收第一组残差与 LayerNorm 的输出，对每个位置分别执行
`Linear(64, 128) → ReLU → Linear(128, 64)`。128 是本教学版本选取的中间维度，
不是位置数，也不是必须使用的固定倍数。ReLU 引入非线性；如果没有非线性，两个线性层
（含偏置）的组合仍可合并成一个线性层。

所有位置共享同一套 FFN 参数，但 FFN 不直接混合不同位置；它处理的是已经由注意力汇总的
上下文特征。输出恢复到 64 维后，加回 FFN 自己的输入，再经过第二个独立的 LayerNorm：

```text
x → 多头注意力 → 加回 x → LayerNorm₁ → y
y → FFN        → 加回 y → LayerNorm₂ → Encoder 输出
```

至此形成一层不含 dropout 的教学版 Post-LN Encoder，输入与输出均为 `[B, 22, 64]`。
输出仍是特征，不是字符概率。`inspect_sequence` 检查 FFN 形状、单位置独立计算、第二组
残差与归一化，以及从最终输出到全部 Encoder 参数和输入序列的梯度；仍不训练或读取权重。
下一步接字符分类头与 CTC。

## 第八步：字符分类头与 CTC

用 `Linear(64, len(characters) + 1)` 把 Encoder 输出转换为 `[B, 22, 21]` 的 logits：
每个时间步有 20 个字符类别和一个 blank 类别。21 不是字符串长度，logits 也尚不是概率。
复用第三阶段 `encode_ctc_targets` 和 `compute_ctc_loss`；后者内部执行 `log_softmax`
并转换为 PyTorch CTC 所需的 `[T, B, C]`，不需要在调用前额外执行 softmax。

示例标签 `a7` 和 `aa2` 编码为拼接的一维目标与长度 `[2, 3]`，无需指定每个字符对应哪个时间步。
CTC 损失汇总能折叠成目标标签的有效路径概率，不是先贪心解码再比较文本。
路径折叠规则为先合并相邻重复类别，再删除 blank：`a,a → a`，`a,blank,a → aa`。
blank 是额外的输出类别，不是真实标签字符，也不是空格或结束符。

`inspect_sequence` 现用随机图片和上述示例标签检查 logits、目标编码、重复字符折叠、有限的
CTC 损失及分类头到 Encoder 输入的梯度。标签不对应随机图片的真实内容，打印的损失仅用于
接口检查，不能视为学习效果；不更新参数，CNN 仍不参与反向传播，未读取数据集或 checkpoint。

## 第九步：可训练模型与极小训练集过拟合

`stage4_transformer_encoder/models.py` 将上述计算整理为 `EncoderLayer` 与 `CaptchaTransformer`。
仍使用显式 Q/K/V 和注意力计算，复用第三阶段 CNN 结构与分类头结构，不加载旧权重。
模型为一层 Post-LN Encoder、4 头、64 维特征、128 维 ReLU FFN、无 dropout，总参数量 141,141。
`inspect_sequence` 现在直接引用模型组件，并校验逐步计算与模型 forward 一致，避免教学与训练实现脱节。

```bash
uv run python -m stage4_transformer_encoder.inspect_sequence
uv run python -m stage4_transformer_encoder.overfit_tiny
```

过拟合检查只读取 `data/stage3_variable_length_v2/train`，按清单顺序从 2～6 位各取前两张，
索引为 `[4,5,2,3,0,1,8,9,12,13]`。这 10 张图片对应 5 个标签的双渲染版本：
`m5`、`bf3`、`x6yp`、`82ffp`、`b36me3`，并非 10 个不同标签。
CPU、4 线程、seed 0、Adam lr=0.001、batch size 10，默认最多 500 轮，全 CNN 与 Encoder
参与反向传播；每轮以 `model.eval()` 在同一批图片上检查整串准确率，达到 100% 即停止。

首轮训练模式 CTC loss 为 16.1703；第 75 轮 loss 为 0.3257、推理模式识别正确 7/10；
第 96 轮 loss 为 0.1360、推理模式正确 10/10，检查通过。包含重复字符的 `82ffp` 同样正确。
这只证明模型能记住极小样本，不能证明泛化能力或超过 BiLSTM。未读取验证集/测试集，未保存权重，
未启动正式训练。后续实验继续沿用第三阶段数据划分与评估口径。

## 第十步：正式训练入口（由学习者运行）

训练命令由学习者亲自运行；AI 只准备代码、执行非训练检查并解读输出，不自行启动训练。

```bash
uv run python -m stage4_transformer_encoder.train --epochs 20
```

默认 `--device auto`，按 CUDA → Apple MPS → CPU 的顺序选择可用设备；可显式指定设备。
CPU 线程数默认 4，batch size 16、Adam lr=0.001、weight decay=0、seed=0。
从随机初始化开始，每轮无放回地遍历 V2 训练集；六位标签的逐样本 CTC loss 权重为 2。
只使用 `data/stage3_variable_length_v2/train` 与原有 `data/stage3_variable_length/validation`，
V2 仅替换训练数据，不另建验证集，也不读取测试集。

沿用第三阶段的训练损失、贪心解码、整串准确率、CER、按长度指标和最佳模型选择逻辑。
每轮在模型副本上用完整训练集重校准 BatchNorm，然后以同一个副本评价训练集和验证集；
优化器引用的原模型保持自己的运行统计，不被校准副本覆盖。校准后验证 loss（无六位加权）
驱动 `ReduceLROnPlateau`，factor=0.3、patience=1、min_lr=1e-5。

最佳模型优先比较验证整串准确率，再比较 CER，最后比较验证 loss。
仅保存到本阶段共享的 `stage4_transformer_encoder/checkpoints/model.pth`，包含校准后的模型状态、
结构与训练配置、最佳轮次和验证指标；原子替换旧文件，不影响第三阶段 checkpoint。
该目录被 Git 忽略。再次运行会重新初始化并覆盖本阶段最佳权重，不是断点续训；
未保存优化器与调度器状态，不支持 `--resume`。

准备阶段只运行 `--help` 与 `test_train_wiring` 等非训练检查。后者使用合成数据、替身训练步骤，
禁止调用 `Adam.step`，检查 BN 副本隔离、参数校验与 checkpoint 保存/严格加载；不读取真实
验证集、不保留临时权重。正式训练效果等待学习者运行后记录。

## 首次正式训练与验证错误分析

学习者在 MPS 上运行 20 轮（batch=16、lr=0.001、seed=0，其余参数沿用上述默认配置）。
最佳为第 19 轮：训练整串准确率 100%、CER 0%，验证整串准确率 90.80%、CER 2.50%、
loss 0.085333，准确率泛化差距 9.20 个百分点。验证 2～6 位准确率依次为
96%、94%、98%、94%、72%。第 15 轮后学习率降至 0.0003；第 20 轮准确率与 CER 未变，
但 loss 略高，未替换第 19 轮权重。

相较第三阶段最佳 BiLSTM（175,573 参数、验证 94.00% / CER 1.60%、六位 84%），
当前 Transformer 为 141,141 参数（少约 19.6%），验证准确率低 3.20 个百分点，
CER 高 0.90 个百分点，六位低 12 个百分点。这是当前配置的单次实验对照，不能推广为架构优劣结论。

只做推理的错误分析命令：

```bash
uv run python -m stage4_transformer_encoder.analyze_validation
```

分析复用第三阶段的 Levenshtein 对齐、统计与报告逻辑，默认自动选设备；未训练、未改权重、
未读取测试集。MPS 复核第 19 轮 checkpoint 得到 227/250 正确，与训练日志一致。
23 个错误字符串合计 25 次字符编辑：替换 23、删除 2、插入 0；其中 21 个字符串只含替换，
2 个只含删除。这里的操作统计来自一条最短编辑对齐，不等于模型内部的因果解释。

六位错 14/50，占全部错误字符串约 60.9%；连续重复字符样本正确 33/39（84.62%），
其 6 个错误中只有 `58a544 → 58a54` 涉及删除，其余均为替换。
另一处删除为 `y5 → 5`，因此不能把所有错误归因于连续重复折叠。
最常见混淆是 `n→m` 3 次、`m→n` 2 次，其余替换对各 1 次。
主要现象是字符替换而非大量漏字；具体是否由图像干扰、特征提取或序列建模导致，仍需进一步检查。

本地报告为 `stage4_transformer_encoder/evaluation_reports/validation_summary.json` 与
`validation_errors.csv`，该目录不提交 Git。

## 验证错误图片抽查

对上述错误明细中的 11 张原图做人工视觉抽查，未改图、未重训：
`nmp`、`nm73`、`4fnnap`、`am`、`47m5be`、`y5`、`58a544`、`a2mmy4`、
`fnxxc`、`8wwgnf`、`b7dygd`。

- `nmp → mmp`、`nm73 → mm73`：首字符 n 附近没有明显干扰线遮挡，字符间也有可见间隔。
  `am → an` 的 m 仍可辨认。这说明至少部分 n/m 错误不能仅用线条遮挡或字符粘连解释。
- `47m5be → 47n5be`：m 的字体笔画粗重，局部形状紧密；`4fnnap → 4fmnap` 中字符大小
  不一致。这些是可见的字体/尺度差异，不能仅凭图像确定哪一个网络模块造成误判。
- `y5 → 5`：斜线穿过 y 的下部；`58a544 → 58a54`：绿色斜线经过末尾重复 4 附近。
  两处漏字都有可见干扰，但尚无无干扰对照，不能断言漏字由干扰线或 CTC 折叠单独导致。
- `a2mmy4 → c2mmw4`：弧线紧邻 y，首字符 a 附近却没有同样遮挡；同一张图的两个替换
  不应强行归为同一个原因。`b7dygd → n7dygd` 的首字符 b 与斜线重叠。
- `fnxxc → pwxxc` 与 `8wwgnf → 8wmgnf` 呈现不同的字体粗细和字形，可见形态差异，
  但并非所有错误位置都被干扰线直接穿过。

抽查支持“错误来源不单一”：既有可见干扰，也有相对清晰的字符混淆。结合训练 100% 与验证
90.80% 的差距，可以确认存在泛化差距，但不能据此定位为 CNN、注意力或位置编码的单独缺陷。
本次未增加层数、时间步、数据或解码器；后续若做正则化对照，应固定其余条件并由学习者运行。

## Dropout 正则化对照（待学习者训练）

新增 `--dropout`，默认 0 保持无 Dropout 基线。此次只在注意力输出投影后、FFN 输出后，
各加一次 Dropout，再做残差相加；不改 CNN、位置编码、注意力权重或 FFN 中间激活。
两次调用虽使用同一个 Dropout 模块，但分别随机生成掩码，原输入的直接残差路径不丢弃。

```text
y = LayerNorm₁(x + Dropout(Attention(x)))
z = LayerNorm₂(y + Dropout(FFN(y)))
```

训练时以概率 p 将分支输出的元素置零，其余元素乘以 `1/(1-p)` 保持期望；
不是删除模型参数或整个字符。推理和 BN 校准时 Dropout 关闭，仍输出完整维度。
它不增加参数，模型仍为 141,141 参数。是否改善验证表现由实验决定，不预先保证。

由学习者运行以下命令，除了 Dropout 外保留首轮实验的 MPS、20 轮及其他配置：

```bash
uv run python -m stage4_transformer_encoder.train --epochs 20 --device mps --dropout 0.1
```

该命令从随机初始化开始，将覆盖本阶段共享 checkpoint，不另存按参数命名的 `.pth`。
基线 90.80% / CER 2.50%、六位 72% 已记录在上文。
checkpoint 记录实际 Dropout 概率，分析入口按该配置构建模型；旧 p=0 权重仍可严格加载。
同一个随机种子不保证两组实验的随机数消耗轨迹完全相同，本轮属于单种子配置对照。

非训练检查覆盖 p=0 默认值、非法概率、训练模式随机性、推理模式与 p=0 的一致性、
校准时关闭 Dropout，以及 p=0.1 配置保存/加载。未启动此次对照训练。

## Dropout 对照结果与位置编码消融

学习者运行残差分支 Dropout=0.1、MPS、20 轮后，最佳为第 20 轮：训练准确率 100%、
CER 0%，验证准确率 90.80%、CER 2.70%、loss 0.0864；2～6 位准确率为
96%、96%、96%、90%、76%。相比无 Dropout 的最佳结果，整串正确数仍为 227/250，
字符编辑错误从 25 增至 27，六位多对 2 张、其他长度合计少对 2 张。
本次单种子对照没有显示总体收益，不能据此断言所有 Dropout 配置均无效。

下一次实验恢复 Dropout=0，只关闭显式正弦位置编码，其他配置保持不变：

```bash
uv run python -m stage4_transformer_encoder.train --epochs 20 --dropout 0 --no-position-encoding
```

默认仍开启位置编码；`--no-position-encoding` 才关闭。关闭只跳过将正弦向量加到 CNN 序列的
步骤，不打乱 22 个横向位置，也不删除 CNN 所提取的空间结构，CTC 仍按原时间轴对齐。
因此该实验检验的是显式位置编码的额外作用，不是宣称整个模型完全没有顺序信息。

直接比较无 Dropout、有位置编码基线的验证 90.80% / CER 2.50%、六位 72%。
这是重新初始化并训练，不是直接在已有权重上关掉位置编码做推理。参数量仍为 141,141。
运行后仍覆盖本阶段唯一共享权重。checkpoint 保存 `use_position_encoding`，分析入口按配置
还原；缺少该字段的历史权重按开启处理。非训练检查覆盖开关默认值、关闭分支计算、形状/参数量
以及配置保存/加载；新对照训练由学习者执行。

## 位置编码消融结果与 Pre-LN 对照

学习者完成关闭位置编码、Dropout=0、MPS、20 轮对照。最佳第 18 轮：训练准确率 100%、
CER 0%，验证准确率 90.80%、CER 2.80%、loss 0.1073，2～6 位准确率为
96%、94%、92%、96%、76%。相比有位置编码基线，整串正确数相同，字符编辑错误从 25 增至 28；
当前没有总体收益，后续恢复位置编码。这不代表无位置编码的自注意力能直接感知数组下标：
CNN 空间结构、序列排列及 CTC 时间轴仍然保留。

下一步研究 LayerNorm 的放置位置。默认仍为 Post-LN，新增 `--pre-norm` 切换至 Pre-LN：

```text
Post-LN: y = LN₁(x + Attention(x)); z = LN₂(y + FFN(y))
Pre-LN:  y = x + Attention(LN₁(x)); z = y + FFN(LN₂(y))
```

Pre-LN 对计算分支的输入做归一化，直接残差路径不经过 LayerNorm；注意力的 Q/K/V 都来自
LN₁(x)。本次只移动层内两个 LayerNorm，不额外增加 Encoder 末尾的最终 LayerNorm，
分类头直接接层输出。因此这是单层内部归一化位置对照，不是所有 Pre-LN 网络设计的比较。
参数量仍为 141,141，不承诺训练或验证表现必然改善。

由学习者运行，恢复位置编码、Dropout=0，其余保持不变：

```bash
uv run python -m stage4_transformer_encoder.train --epochs 20 --dropout 0 --pre-norm
```

不加 `--no-position-encoding`；自动选设备时确认仍为 MPS。与有位置编码、无 Dropout 的
Post-LN 基线 90.80% / CER 2.50%、六位 72% 对照，观察 loss 波动、降学习率时点和最佳验证指标。
此命令从头训练并覆盖共享权重；`norm_first` 随 checkpoint 保存，分析入口按配置还原。
非训练检查使用相同参数与 PyTorch `TransformerEncoderLayer` 对照两种前向计算，并验证
计算分支置零后 Pre-LN 残差保留输入，以及配置保存/加载；未启动此次训练。

## 缩小 FFN 容量对照（待学习者训练）

新增 `--ffn-dim`，默认 128 保持原模型；本次只将 FFN 中间维度从 128 改为 64，其他条件恢复基线：
Post-LN、位置编码开启、Dropout=0。

```bash
uv run python -m stage4_transformer_encoder.train --epochs 20 --dropout 0 --ffn-dim 64
```

FFN 由 `64 → 128 → 64` 改为 `64 → 64 → 64`，参数量减少 8,256（141,141 → 132,885）；
注意力、CNN、时间步和 CTC 不变。请比较训练/验证差距、验证 CER 和六位准确率；训练仍由学习者运行。

## 只用 V2 新渲染对照（待学习者训练）

新增 `--new-renders-only`，只保留 V2 manifest 中 `source=new_render` 的 1,500 张训练图，
验证集和模型配置不变。这是诊断对照，不是公平的增量数据实验，因为训练样本量减半。

```bash
uv run python -m stage4_transformer_encoder.train --epochs 20 --dropout 0 --new-renders-only
```

保持 Post-LN、位置编码、FFN=128；请先确认日志显示 `训练=1500`。不预设准确率会提升，
训练仍由学习者运行并覆盖共享 checkpoint。

## 在线视觉多样性：同标签独立重渲染（待学习者训练）

已完成的仿射增强 40 轮对照：seed=1 最佳第 37 轮，验证准确率 95.20%、CER 1.30%、
六位 88%；seed=2 最佳仍为第 16 轮，验证准确率 94.00%、CER 1.70%、六位 80%。
两个种子的均值为准确率 94.60%、CER 1.50%、六位 84%。延长训练并非每个种子都有收益。

新增 `--rerender-probability`（默认 0，关闭）。本次设为 0.5：每次读取训练样本时，
50% 概率保留原图，50% 概率用原标签重新生成图片，然后统一执行 `--augment` 的轻微仿射增强。
复用第三阶段生成器，字体从当前训练记录中出现过的字体去重后均匀抽取；颜色、背景、干扰与
字形扰动重新取样。只在内存生成，不增加标签、不改磁盘数据，每轮仍是 3,000 个样本。
重渲染使用独立的 Python 随机流，当前单进程加载下同种子、同读取顺序可复现；
概率为 0 时不消耗该随机流，保留原有仿射行为。启用时字体缺失会报错，不静默换字体。

验证、原图训练指标和 BN 校准仍读取原图，不重渲染、不仿射；不读取测试集。
因此训练准确率是对固定训练原图的评价，并非当轮所有随机视图的准确率。
本次只检验增加视觉变化是否有收益，不预设一定提高准确率。与上述同种子 40 轮结果比较：

```bash
for seed in 1 2; do
  uv run python -m stage4_transformer_encoder.train \
    --epochs 40 --dropout 0 --augment \
    --rerender-probability 0.5 --seed "$seed" || break
done
```

仍从随机初始化开始，自动选择设备，依次覆盖本阶段唯一共享 checkpoint；记录重渲染概率。
CPU 在线绘图会增加训练耗时，不生成额外 PNG 或分种子权重。训练由学习者运行。
非训练检查命令：`uv run python -m stage4_transformer_encoder.test_train_wiring`，
覆盖开关、混合分支、同种子复现、标签/形状、原数据不变及 BN/验证隔离；禁止优化器更新。

## 保留横向细节：22 → 45 个位置（待学习者训练）

50% 在线重渲染的 40 轮结果：seed=1 最佳第 20 轮，准确率 95.60%、CER 1.20%、六位 84%；
seed=2 最佳第 32 轮，准确率 94.40%、CER 1.40%、六位 82%。两个种子平均准确率 95.00%、
CER 1.30%、六位 83%，相对仅仿射增强是小幅收益，不是六位准确率的稳定提升。
当前 seed=2 权重验证复现为 14 张错误：10 次替换、3 次删除、1 次插入；其中六位 9 张
（7 次替换、2 次删除）。n→m 四次、m→n 一次。提示需要检验字形细节辨别，并不证明下采样是根因。

新增 `--sequence-width 45`，默认仍为 22。复用第三阶段的池化配置：第三次池化从 `(2, 2)`
改成 `(2, 1)`，只减少横向下采样。输入仍为 180×100，特征序列从 `[B,22,64]` 变为
`[B,45,64]`；正弦位置编码按实际长度生成，每个注意力头仍看所有位置，CTC 自动使用输出长度。
参数量仍为 141,141，但注意力矩阵从 22×22 变为 45×45，计算和内存开销增加。
这也改变了横向感受野及 CTC 对齐空间，并非只改变细节清晰度；不预设一定改善。

其余设置保持本轮基线，由学习者运行：

```bash
for seed in 1 2; do
  uv run python -m stage4_transformer_encoder.train \
    --epochs 40 --dropout 0 --augment --rerender-probability 0.5 \
    --sequence-width 45 --seed "$seed" || break
done
```

从头训练并覆盖共享 checkpoint，不直接复用 22 位置权重做对照。权重保存实际 sequence_width，
验证分析入口按配置还原。非训练检查覆盖默认 22、45 位置前向与 CTC loss、参数量不变、
45 位置 checkpoint 保存/加载及验证分析入口；不启动训练、不读取测试集。

## 45 位置结果与解码对照

40 轮、仿射增强、重渲染概率 0.5：seed=1 最佳第 30 轮，验证准确率 96.00%、CER 1.10%、
六位 88%；seed=2 最佳第 39 轮，准确率 96.40%、CER 1.00%、六位 90%。
两种子均值准确率 96.20%、CER 1.05%、六位 89%，优于相同设置的 22 位置结果
（95.00%、1.30%、83%），但仅两个种子、250 张验证图，不作统计显著性结论。

对当前 seed=2、第 39 轮权重做不训练解码对照：

```bash
uv run python -m stage4_transformer_encoder.analyze_validation --decoder beam --beam-width 10
```

beam=10 与 greedy 均为 241/250 正确、CER 1.00%、六位 90%；逐错误 CSV 完全一致，
9 张错图合计 10 次编辑（5 替换、2 删除、3 插入），两个报告使用同一份 checkpoint。
本次没有解码收益，继续保留 greedy；不能推断所有 beam 宽度均无效，也不能仅凭此断定错误根因。
复用第三阶段 prefix beam search，不加词典或长度约束。默认仍 greedy，beam 报告写入
`stage4_transformer_encoder/evaluation_reports/beam_10/`，不覆盖 greedy 报告或权重。
非训练接线检查覆盖 beam 参数传递和报告目录；测试集未读取。

## 第四阶段收尾：已完成

学习者确认当前效果满足本阶段学习目标，结束 CNN + Transformer Encoder + CTC 阶段，
不再追加调参或实验。上文“待学习者训练”标题保留为历史实验计划，结果以后续记录及本节为准。

- 最终采用配置：45 个横向位置、64 维特征、单层四头 Encoder、FFN=128、Post-LN、
  正弦位置编码、Dropout=0；参数量 141,141。
- 训练配置：40 轮、batch=16、初始 lr=0.001、六位 loss 权重=2、轻微仿射增强、
  在线同标签重渲染概率=0.5、BN 原图重校准、自动学习率；推理保留 greedy。
  CLI 默认 sequence_width 仍为 22，复现最终配置需显式传入 `--sequence-width 45`。
- 最终共享权重：seed=2、第 39 轮；验证集 241/250 正确（96.40%），CER=1.00%，
  六位准确率 90%。seed=1/2 最佳验证准确率均值为 96.20%，不是测试集或生产准确率。
- 学习覆盖：CNN 特征转序列、位置编码、Q/K/V、多头注意力、残差连接、LayerNorm、
  Pre/Post-LN、FFN、CTC 对齐与解码，以及控制变量、多种子对照和错例分析。
- 实验结论：本项目中视觉多样性与减少横向下采样获得收益；Dropout、缩小 FFN 等没有
  显示预期的总体收益；当前权重 beam=10 与 greedy 完全一致。以上结论仅适用于已测配置。

保留本阶段唯一共享 checkpoint，不额外复制权重，不提交训练生成的 `.pth`。
本次收尾只更新文档，不训练、不读取测试集；独立测试评估尚未执行。
