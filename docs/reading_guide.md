# 怎样读回自己的代码

这次整理保留你的 CNN、训练与验证循环、中文注释和纯 state_dict 权重。没有增加公共工具模块、训练框架或新 checkpoint 格式。

不用从 train.py 第一行开始背。先按下面的顺序读。

## 第一次：只看 network.py

看 __init__：它是在搭网络。再看 forward：它是在规定图片怎么走。

```text
图片 [N,1,28,28]
→ conv1 [N,16,14,14]
→ conv2 [N,32,7,7]
→ flatten [N,1568]
→ fc [N,10]
```

这里的 N 是 batch 大小。最后 10 个数是类别分数 logits，训练时直接交给 CrossEntropyLoss。

能回答“为什么全连接层输入是 32×7×7”，这一遍就完成了。

## 第二次：只看 train.py 里的训练循环

找到 for images, labels in train_loader，重点看你原来写的几行：

```python
optimizer.zero_grad()          # 清掉上一批梯度
outputs = model(images)        # 图片经过 CNN，得到类别分数
loss = criterion(outputs, labels)  # 比较预测与真实标签
loss.backward()               # 算各参数的梯度
optimizer.step()              # 按梯度更新参数
```

取数据是循环本身，所以这是“取数据 + 五行操作”。它们的含义和你的原版本相同。

这一遍只解释清楚：哪些行在预测，哪些行在更新参数。准确率累计和日志先跳过。

## 第三次：训练与验证的差别

训练前 model.train()；验证前 model.eval()。验证在 torch.no_grad() 中，计算 loss 和 accuracy，但不执行 backward 和 optimizer.step。

然后看 if val_loss < best_val_loss：验证变好就保存 deepcopy 出来的权重；连续 3 轮没变好就早停。deepcopy 的原写法保留了，避免“最佳权重”跟着后续训练变动。

## 第四次：回头看数据

load_data.py 里的 get_dataset 只是一层很薄的函数：原先导入模块就直接创建 Dataset，现在在你需要时调用。

```python
train_dataset = get_dataset(train=True, augment=True)
val_dataset = get_dataset(train=True, augment=False)
```

train=True 表示读取官方训练数据。augment 才表示是否增强。验证集也是从官方训练集划出，所以这两行都使用 train=True。

真正的 50,000/10,000 划分仍写在 train.py，仍是你的 randperm + Subset。验证集索引与训练集不重叠。

## 新增部分可以晚一点读

| 新增部分 | 在做什么 | 什么时候看 |
| --- | --- | --- |
| argparse 与 --output | 运行时选择模型路径、种子和是否增强 | 能解释训练循环以后 |
| random/NumPy/torch 的 seed | 让重跑更容易对照 | 做增强对照时 |
| CSV / JSON 写入 | 留下每轮指标和训练参数 | 准备汇报结果时 |
| test.py 的 save-dir | 将已有测试结果存成文件 | 想把图放进报告时 |
| main() 与文件末尾的判断 | 直接运行文件才执行；被导入时不自动训练、下载或弹窗 | 理解函数之后 |

你可以先把新增记录部分当作“把结果写进文件”，不用一开始就逐行掌握文件操作。

读完后，能用自己的话讲出“训练用什么数据、验证用来做什么、哪一轮权重被保存、测试有没有更新参数”，就已经能清楚介绍这个项目。
