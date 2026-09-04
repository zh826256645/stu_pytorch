# stu-pytorch

一个用于学习 PyTorch 计算机视觉模型的固定长度验证码识别项目。项目按两个阶段推进：

1. **第一阶段：使用 DenseNet121 初尝 CNN 识别**——借助 ImageNet 预训练模型，从整张图片直接预测 4 个字符。
2. **第二阶段：实现一个简单 CNN**——使用 `nn.Conv2d` 等基础组件自己搭建网络，理解卷积网络的结构和训练过程。

## 第一阶段：DenseNet121

当前第一阶段使用 ImageNet 预训练的 DenseNet121，从整张图片直接预测 4 个字符，
不进行字符切割。

### 当前模型

- 输入：RGB 图片，统一缩放为 `180 × 100`
- 配置字符集：`0-9` 和 `a-z`，共 36 类
- 输出：`4 × 36 = 144` 个分数
- 训练集：`data/train`，800 张
- 验证集：`data/test`，381 张
- 权重：`stage1_densenet121/checkpoints/model-36.pth`
- 准确率：4 个字符全部正确才计为一次正确

当前数据文件名均为 4 个字符，例如 `25mb.png`。文件名（不含扩展名）就是监督
标签。当前数据实际只覆盖 `2345678abcdefgmnpwxy` 这 20 个字符，因此训练出的
权重不能视为可靠支持全部 36 类。数据目录中只能放符合该规则的图片文件。

## 环境准备

项目需要 Python 3.13 和 [uv](https://docs.astral.sh/uv/)：

以下命令都需要在项目根目录执行，因为数据和权重使用相对路径。

```bash
uv sync
```

项目使用 `uv.lock` 固定依赖版本。

macOS 运行 Tkinter GUI 还需要：

```bash
brew install python-tk@3.13
```

PyTorch 会自动选择可用设备，顺序为 CUDA、Apple MPS、CPU。默认首次训练会下载
DenseNet121 的 ImageNet 预训练权重，之后使用本机缓存；使用 `--no-pretrained` 时不会下载。

## 训练

当前推荐基线：

```bash
uv run python -m stage1_densenet121.train --loss multi-label --lr 0.001 --backbone-lr 0.0003
```

训练 30 轮，每轮评估一次，只将验证准确率最高的权重保存到
`stage1_densenet121/checkpoints/model-36.pth`；准确率相同时保留 loss 更低的权重。

可用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--loss` | `multi-label` | `multi-label` 或 `cross-entropy` |
| `--lr` | `0.001` | 分类层学习率；不设置主干学习率时也是全模型学习率 |
| `--backbone-lr` | 未设置 | 单独指定 DenseNet 特征主干学习率 |
| `--classifier-only` | 关闭 | 冻结 DenseNet 特征层，只训练分类层 |
| `--no-pretrained` | 关闭 | 不加载 ImageNet 预训练权重，从随机初始化开始训练 |
| `--plot-curves` | 关闭 | 实时显示训练集/验证集的 loss 和整张验证码准确率曲线 |

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

启用 `--plot-curves` 后会打开一个 Matplotlib 窗口，每轮结束后更新两张图：
训练集/验证集 loss，以及训练集/验证集的整张验证码准确率。训练结束后关闭图表窗口，程序才会退出。

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

## 冒烟测试

```bash
uv run python -m stage1_densenet121.test_smoke
```

该检查会验证数据读取、模型的 144 维输出，以及两种损失函数能否正常计算；不会
修改训练权重。

## 文件结构

| 路径 | 用途 |
| --- | --- |
| `datasets.py` | 两个阶段共用：从文件名构造标签并加载图片 |
| `stage1_densenet121/models.py` | 第一阶段 DenseNet121 和 144 维分类层 |
| `stage1_densenet121/train.py` | 第一阶段训练、验证和最佳权重保存 |
| `stage1_densenet121/predict.py` | 第一阶段单张识别与测试集逐张预测 |
| `stage1_densenet121/main.py` | 第一阶段 Tkinter 桌面界面 |
| `stage1_densenet121/test_smoke.py` | 第一阶段最小可运行检查 |
| `stage1_densenet121/checkpoints/model-36.pth` | 第一阶段 DenseNet121 权重 |
| `data/train2`、`data/test2` | 旧算式验证码数据，当前训练不使用 |
| `checkpoints/model.pth` | 旧算式验证码权重，当前代码不使用 |

第二阶段的简单 CNN 将在后续加入独立目录，复用根目录下的数据加载逻辑。

`get_imgs.py` 是旧算式验证码的下载脚本，当前 36 类训练流程不使用。

## 已知限制

- 只能识别固定 4 个字符。
- 标签必须来自图片文件名。
- 当前 `data/test` 同时承担最佳模型选择，严格来说属于验证集，没有独立测试集。
- GUI 每次识别都会重新加载模型，适合演示，不适合高吞吐服务。
- 输入只执行 `ToTensor()`，尚未使用 ImageNet mean/std 标准化。
- 仅应在自有或明确授权的验证码系统中使用。
