# 第一阶段：DenseNet121

[返回项目首页](../../README.md) · [环境准备](../../README.md#环境准备)

本文所有命令和路径均以项目根目录为基准。

当前第一阶段使用 ImageNet 预训练的 DenseNet121，从整张图片直接预测 4 个字符，
不进行字符切割。

## 当前模型

- 输入：RGB 图片，统一缩放为 `180 × 100`
- 配置字符集：`0-9` 和 `a-z`，共 36 类
- 输出：`4 × 36 = 144` 个分数
- 训练集：`data/train`，800 张
- 验证集：`data/test`，381 张
- 权重：`stage1_densenet121/checkpoints/model-36.pth`
- 整图准确率：4 个字符全部正确的图片数 ÷ 图片总数
- 单字符准确率：预测正确的字符位置数 ÷ 字符位置总数

当前数据文件名均为 4 个字符，例如 `25mb.png`。文件名（不含扩展名）就是监督
标签。当前数据实际只覆盖 `2345678abcdefgmnpwxy` 这 20 个字符，因此训练出的
权重不能视为可靠支持全部 36 类。数据目录中只能放符合该规则的图片文件。

## 训练

当前推荐基线：

```bash
uv run python -m stage1_densenet121.train --loss multi-label --lr 0.001 --backbone-lr 0.0003
```

训练 30 轮，每轮评估一次，同时输出整图准确率和单字符准确率。只将验证整图准确率
最高的权重保存到 `stage1_densenet121/checkpoints/model-36.pth`；整图准确率相同时保留
loss 更低的权重。

两个阶段共用的 `--epochs`、`--batch-size`、`--lr` 和 `--plot-curves` 由
`training_config.py` 中的 `TrainingConfig` 统一定义和校验。两个阶段可以分别设置默认值。
曲线绘制逻辑统一放在 `training_curves.py` 中。

可用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | `30` | 训练轮数 |
| `--batch-size` | `5` | 每批图片数量 |
| `--loss` | `multi-label` | `multi-label` 或 `cross-entropy` |
| `--lr` | `0.001` | 分类层学习率；不设置主干学习率时也是全模型学习率 |
| `--backbone-lr` | 未设置 | 单独指定 DenseNet 特征主干学习率 |
| `--classifier-only` | 关闭 | 冻结 DenseNet 特征层，只训练分类层 |
| `--no-pretrained` | 关闭 | 不加载 ImageNet 预训练权重，从随机初始化开始训练 |
| `--plot-curves` | 关闭 | 实时显示训练集/验证集的 loss、整图准确率和单字符准确率曲线 |

示例：

```bash
# 只训练最后一层
uv run python -m stage1_densenet121.train --classifier-only --loss multi-label --lr 0.001

# 主干和分类层使用不同学习率
uv run python -m stage1_densenet121.train --loss multi-label --lr 0.001 --backbone-lr 0.0003

# 不使用 ImageNet 预训练权重，从随机初始化开始训练
uv run python -m stage1_densenet121.train --no-pretrained --loss multi-label --lr 0.001

# 实时显示训练曲线
uv run python -m stage1_densenet121.train --plot-curves --loss multi-label --lr 0.001
```

启用 `--plot-curves` 后会打开一个 Matplotlib 窗口，每轮结束后更新三张图：
训练集/验证集 loss、整图准确率和单字符准确率。训练结束后关闭图表窗口，程序才会退出。

使用推荐的分层学习率时，一次本机参考实验达到 `99.21%`，即 381 张验证图片中
正确 378 张。该结果仅反映当前数据集，不代表其他验证码来源的识别率。

## 预测

训练完成后运行：

```bash
uv run python -m stage1_densenet121.predict
```

程序会逐张打印预测值和真实标签，并显示对应图片。关闭当前图片窗口后才会继续
下一张。

## GUI

```bash
uv run python -m stage1_densenet121.main
```

点击“选择图片”，选择一张文件名为正确标签的图片，再点击“识别”。GUI 的累计
准确率依赖文件名标签；文件名不是 4 个合法字符时会抛出 `ValueError`。

GUI 每次识别都会重新加载模型，适合演示，不适合高吞吐服务。

## 冒烟测试

```bash
uv run python -m stage1_densenet121.test_smoke
```

该检查会验证数据读取、模型的 144 维输出，以及两种损失函数能否正常计算；不会
修改训练权重。

## 文件结构

| 路径 | 用途 |
| --- | --- |
| `stage1_densenet121/models.py` | 第一阶段 DenseNet121 和 144 维分类层 |
| `stage1_densenet121/train.py` | 第一阶段训练、验证和最佳权重保存 |
| `stage1_densenet121/predict.py` | 第一阶段单张识别与测试集逐张预测 |
| `stage1_densenet121/main.py` | 第一阶段 Tkinter 桌面界面 |
| `stage1_densenet121/test_smoke.py` | 第一阶段最小可运行检查 |
| `stage1_densenet121/checkpoints/model-36.pth` | 第一阶段 DenseNet121 权重 |
