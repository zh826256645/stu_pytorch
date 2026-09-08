# stu-pytorch

一个用于学习 PyTorch 计算机视觉模型的验证码识别项目。项目按四个阶段推进：

1. [第一阶段：使用 DenseNet121 初尝 CNN 识别](docs/stages/01-densenet121.md)——借助 ImageNet 预训练模型，从整张图片直接预测 4 个字符。
2. [第二阶段：实现一个简单 CNN](docs/stages/02-simple-cnn.md)——使用 `nn.Conv2d` 等基础组件自己搭建网络，理解卷积网络的结构和训练过程。
3. [第三阶段：实现可变长度验证码识别](docs/stages/03-cnn-ctc.md)——使用 CNN 提取横向视觉特征，结合 BiLSTM 与 CTC 在无需字符切分的情况下识别 2～6 位验证码，理解序列建模与自动对齐。
4. [第四阶段：CNN + Transformer Encoder + CTC（已完成）](docs/stages/04-transformer-encoder-ctc.md)——在相同数据上学习自注意力序列建模，逐步替换 BiLSTM。

各阶段文档完整保留教程、训练参数、实验结果和结论；新增实验记录写入对应阶段文档。

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

PyTorch 会自动选择可用设备，顺序为 CUDA、Apple MPS、CPU。第一阶段默认首次训练会下载
DenseNet121 的 ImageNet 预训练权重，之后使用本机缓存；使用 `--no-pretrained` 时不会下载。

## 代码格式与检查

项目使用 Ruff 统一检查和格式化 Python 代码。Ruff 作为开发依赖记录在 `pyproject.toml`
和 `uv.lock` 中，执行 `uv sync` 时会一并安装。

```bash
# 检查代码问题
uv run ruff check .

# 自动修复可以安全修复的问题
uv run ruff check . --fix

# 格式化代码
uv run ruff format .

# 只检查格式，不修改文件
uv run ruff format --check .
```

提交代码前推荐运行：

```bash
uv run ruff check . && uv run ruff format --check .
```

当前启用了常见代码错误、未使用代码、import 排序、现代 Python 写法和 bug-prone
模式检查，目标 Python 版本为 3.13，默认行宽为 88。

## 文件结构

下列共用模块服务于第一、第二阶段；各阶段专属文件见对应阶段文档。

| 路径 | 用途 |
| --- | --- |
| `datasets.py` | 两个阶段共用：从文件名构造标签并加载图片 |
| `training_config.py` | 两个阶段共用：训练参数类和命令行参数定义 |
| `training_curves.py` | 两个阶段共用：实时 loss、整图准确率和单字符准确率曲线 |
| `training_metrics.py` | 两个阶段共用：整图与单字符准确率统计 |
| `data/train2`、`data/test2` | 旧算式验证码数据，当前训练不使用 |
| `checkpoints/model.pth` | 旧算式验证码权重，当前代码不使用 |

`get_imgs.py` 是旧算式验证码的下载脚本，当前 36 类训练流程不使用。

## 使用范围

仅应在自有或明确授权的验证码系统中使用。
