# PyTorch CNN Learning

用 PyTorch 完成 MNIST 手写数字分类，记录从数据读取、CNN 训练到错误分析的学习过程。

已实现：50,000/10,000 训练与验证划分、CNN、Early Stopping、最佳权重保存、独立测试脚本、分类报告、混淆矩阵、错图分析、单张图片预测及卷积可视化。训练循环保留展开写法与中文注释。

2026-10-02 重新评估现有权重：原模型 **98.90%**，增强版 **99.13%**，测试集均为 MNIST 官方 10,000 张图片。这是两份历史权重的结果；原训练种子和过程记录不完整，尚不能据此认定提升完全来自增强。

2026-10-06 完成 seed=42、43、44 三组配对实验：原图平均准确率从 **98.70%** 提高至 **99.12%**，平均配对提升 **0.42 个百分点**；同时报告固定 ±10° 旋转测试。见 [完整复验报告](docs/results/repeated_seeds42_44/README.md)。

## 从哪里开始读

先读 [代码阅读指南](docs/reading_guide.md)，再看 `network.py` 和 `train.py` 的训练循环。参数解析和文件记录可以先跳过。

- [MNIST 使用说明](mnist/README.md)：安装、运行、对照实验与结果。
- [逐文件审查](docs/code_review.md)：原问题、修改点、写得好的部分和导师展示准备。

## 运行

在仓库根目录创建并激活环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python mnist/train.py
python mnist/test.py --no-plot
python mnist/predict.py
```

验证环境为 Windows、Python 3.13.7、PyTorch 2.14.0 / torchvision 0.29.0、CPU。依赖按本次验证版本固定。CUDA 用户先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装相匹配的 torch/torchvision，再安装其余依赖；本次没有验证 GPU。

权重和数据不提交，新克隆的仓库需要先训练。训练默认启用增强，保存到 `mnist/models/best_model_augmented.pth`。已存在的实验文件会被保护，请通过 `--output` 使用新文件名。

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
│   ├── show_augmentation.py
│   ├── visualize_features.py
│   ├── visualize_kernels.py
│   ├── images/test.png
│   ├── models/                  # 本地权重、训练 csv/json
│   ├── data/.gitkeep            # 早期目录占位，保留学习痕迹
│   └── README.md
└── docs/
    ├── reading_guide.md
    ├── code_review.md
    └── results/                 # 已核验的展示材料
```

## 后续学习路线

已建立 [CIFAR-10 基础框架](cifar10/README.md)，从数据读取开始自行编写代码。

完成增强对照与自写数字测试 → CIFAR-10 → BatchNorm / Dropout → ResNet → 迁移学习。后续项目尚未实现。
