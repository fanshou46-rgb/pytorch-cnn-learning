# PyTorch CNN Learning

这是一个逐步学习 PyTorch 图像分类的仓库。MNIST 项目已完成数据读取、CNN 训练、验证、早停、独立测试和错误分析；CIFAR-10 已建立基础框架，代码由学习者继续编写。

## 当前进度

| 项目 | 状态 | 内容 |
| --- | --- | --- |
| [MNIST](mnist/README.md) | 已实现并完成 CPU 实验 | 数字分类、训练增强、最佳权重保存、分类报告、错图分析、卷积可视化、固定旋转测试 |
| [CIFAR-10](cifar10/README.md) | 基础框架 | load_data.py、network.py、train.py、test.py 均为空，下一步从数据读取开始 |

代码保留展开的训练/验证循环和中文注释。数据、模型权重和日常输出保存在本地，已核验的实验记录与展示图片保存在 docs/results/。

## 从哪里开始读

先读 [代码阅读指南](docs/reading_guide.md)，再看 `network.py` 和 `train.py` 的训练循环。参数解析和文件记录可以先跳过。

- [MNIST 使用说明](mnist/README.md)：安装、运行、对照实验与结果。
- [逐文件审查](docs/code_review.md)：原问题、修改点、写得好的部分和导师展示准备。

## 运行

以下命令在仓库根目录执行。先创建并激活环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

验证环境为 Windows、Python 3.13.7、PyTorch 2.14.0+cpu、torchvision 0.29.0+cpu。依赖按本次验证版本固定，见 [requirements.txt](requirements.txt)。CUDA 环境可根据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)选择对应版本；本仓库的实验验证使用 CPU。

也可以直接使用虚拟环境解释器，例如 `.\.venv\Scripts\python.exe mnist/network.py`。首次读取 MNIST 会自动下载到根目录 data/。新克隆的仓库需要先训练，或自行提供结构匹配的权重。

### 训练与原图测试

```powershell
python mnist/train.py --no-augmentation --seed 42 --output mnist/models/baseline_seed42.pth
python mnist/train.py --seed 42 --output mnist/models/augmented_seed42.pth
python mnist/test.py --model mnist/models/baseline_seed42.pth --no-plot --save-dir mnist/results/baseline_seed42
python mnist/test.py --model mnist/models/augmented_seed42.pth --no-plot --save-dir mnist/results/augmented_seed42
```

训练参数：50,000 张训练、10,000 张验证，batch_size=64，Adam(lr=0.001)，最多 20 轮，验证损失连续 3 轮没有改善时早停。训练增强为 RandomRotation(10) 和 RandomAffine(degrees=0, translate=(0.1, 0.1))；验证和原图测试只做 ToTensor()。

训练生成同名 .pth 权重、.csv 每轮指标和 .json 配置。已有训练文件会被保护，重跑时通过 --output 指定新文件名。测试会保存准确率、损失、分类报告、混淆矩阵、逐张预测和高置信度错图。--no-plot 关闭绘图窗口，同时指定 --save-dir 仍会保存图片；测试输出请使用新的目录保存每次实验。

seed=43、44 的运行方式相同：同时修改 --seed 和模型/结果路径。

### 固定旋转测试

```powershell
python mnist/test_rotation.py --baseline mnist/models/baseline_seed42.pth --augmented mnist/models/augmented_seed42.pth --save-dir mnist/results/rotation_seed42
```

固定测试 −10°、0°、+10°，使用最近邻插值、黑色填充，画布保持 28×28。结果为 metrics.csv 与含权重 SHA-256 的 config.json。已有结果会被保护，重跑时指定新的 --save-dir。docs/results/ 保存已提交的记录，新实验建议输出到本地 mnist/results/。

### 单张预测与可视化

```powershell
python mnist/predict.py --model mnist/models/augmented_seed42.pth --image mnist/images/test.png
python mnist/show_augmentation.py --index 0
python mnist/visualize_features.py --model mnist/models/augmented_seed42.pth --index 0
python mnist/visualize_kernels.py --model mnist/models/augmented_seed42.pth
```

单张预测使用灰度化、resize 28×28 和 ToTensor()，白底黑字图片添加 --invert。输入建议使用居中的单个数字，四周留白。特征图展示两层 ReLU 后、池化前的输出；各通道图像独立缩放颜色，适合观察形状。

## MNIST 实验结果

2026-10-06 完成 seed=42、43、44 三组配对实验。每个种子内，两组使用相同的初始化、训练/验证划分、数据顺序和训练参数；区别是训练时是否开启随机旋转和平移。跨种子会同时改变这些随机设置。所有模型按验证损失选择最佳权重，在同一份官方 10,000 张测试图片上评估。

| seed | 基线准确率 | 增强准确率 | 提升（百分点） |
| --- | ---: | ---: | ---: |
| 42 | 98.86% | 99.22% | +0.36 |
| 43 | 98.62% | 99.19% | +0.57 |
| 44 | 98.62% | 98.96% | +0.34 |
| 平均值 | 98.70% | 99.12% | +0.42 |

两组原图准确率的样本标准差均约为 0.14 个百分点，配对提升的样本标准差约为 0.13 个百分点。统计来自这三组实验。

固定旋转测试中，每张原图按相同角度旋转，再分别输入两个模型，参数保持不变：

| 固定角度 | 基线平均准确率 | 增强平均准确率 | 平均提升（百分点） |
| --- | ---: | ---: | ---: |
| −10° | 97.24% | 98.15% | +0.91 |
| 0° | 98.70% | 99.12% | +0.42 |
| +10° | 97.69% | 98.16% | +0.46 |

在三个种子和这两个旋转角度下，增强模型的准确率都更高。旋转后的相对降幅因种子和方向而异；完整报告同时列出了全部结果和标准差。增强方案包含旋转和平移，实验结果反映整个方案的效果。

- [完整配对实验报告](docs/results/repeated_seeds42_44/README.md)
- [逐种子、逐角度数据](docs/results/repeated_seeds42_44/all_results.csv)
- [均值、标准差与最佳训练轮次](docs/results/repeated_seeds42_44/summary.json)
- [seed=42 原图错误分组](docs/results/comparison_seed42/summary.json)
- [+10° 原图与旋转图对照](docs/results/rotation_seed42/plus10_errors/comparison.png)

历史权重的 98.90% / 99.13% 测试结果保存在 docs/results/legacy_baseline/ 和 docs/results/augmented/，其训练配置记录不完整。历史结果与上述配对实验分别报告，详情见 [MNIST 说明](mnist/README.md#已核验的历史结果)。

## 目录

```text
pytorch-cnn-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                       # 自动下载的缓存，本地保留，Git 忽略
├── cifar10/                    # 新项目基础框架，Python 文件待自己实现
│   ├── load_data.py
│   ├── network.py
│   ├── train.py
│   ├── test.py
│   ├── models/
│   ├── results/
│   └── README.md
├── mnist/
│   ├── load_data.py             # 数据和增强
│   ├── network.py               # CNN 定义
│   ├── train.py                 # 训练、验证、早停
│   ├── test.py                  # 测试与错图分析
│   ├── test_rotation.py         # 固定旋转下比较两份模型
│   ├── predict.py               # 自己的手写图片预测
│   ├── show_augmentation.py     # 同一原图的随机增强展示
│   ├── visualize_features.py    # 两层特征图
│   ├── visualize_kernels.py     # 第一层卷积核
│   ├── images/test.png          # 自写数字示例
│   ├── models/.gitkeep          # 本地权重与训练记录的目录占位
│   ├── data/.gitkeep            # 早期目录占位，保留学习痕迹
│   └── README.md
└── docs/
    ├── reading_guide.md
    ├── code_review.md
    └── results/                 # 已提交的实验记录和展示图片
```

运行后会产生根目录 data/、本地 .venv/、mnist/models/ 中的权重与训练日志，以及指定的测试输出目录。数据、权重和日常输出由 .gitignore 管理；用于汇报的实验记录保存在 docs/results/。

## 下一步学习

从 cifar10/load_data.py 开始：读取一批 CIFAR-10 图片，检查图片与标签形状、观察样本，再逐步完成网络、训练和测试。详细顺序见 [CIFAR-10 学习说明](cifar10/README.md)。后续再学习 BatchNorm、Dropout、ResNet 与迁移学习。
