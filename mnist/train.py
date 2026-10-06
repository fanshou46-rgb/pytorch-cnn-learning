"""Train the MNIST CNN with augmentation and clean validation images."""
import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from load_data import get_dataset
from network import CNN


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="训练 MNIST CNN，验证集用于早停。")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0 or not 0 <= args.seed < 2**32:
        parser.error("epochs 必须大于 0；seed 必须在 0 到 2**32-1 之间。")

    model_path = args.output or here / "models" / (
        "best_model_baseline.pth" if args.no_augmentation else "best_model_augmented.pth"
    )
    if model_path.suffix not in (".pth", ".pt"):
        parser.error("output 请使用 .pth 或 .pt 后缀。")
    history_path = model_path.with_suffix(".csv")
    config_path = model_path.with_suffix(".json")
    # 不覆盖旧实验；需要重跑时，请用 --output 指定新的文件名。
    if any(path.exists() for path in (model_path, history_path, config_path)):
        raise FileExistsError(f"实验文件已存在，请更换 --output：{model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 同时固定模型初始化、数据增强和随机划分所用的随机数。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    train_dataset = get_dataset(train=True, augment=not args.no_augmentation)
    val_dataset = get_dataset(train=True, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备: ", device)#选择设备，优先使用GPU，如果没有GPU则使用CPU

    #划分训练集和验证集
    train_size = 50000
    val_size = 10000

    assert train_size + val_size == len(train_dataset)
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_data = Subset(train_dataset, indices[:train_size])
    val_data = Subset(val_dataset, indices[train_size:train_size + val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))#训练集数据加载器
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False,num_workers=0)#验证集数据加载器

    model = CNN().to(device)#实例化CNN模型并将其移动到设备上

    criterion = nn.CrossEntropyLoss()#定义损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#定义优化器

    #early stopping参数
    best_val_loss = float('inf')#初始化最佳验证损失为无穷大
    patience = 3
    wait =0
    best_model_state = None#保存最佳模型权重
    history = []  # 每一轮指标，方便画学习曲线、检查过拟合

    #训练参数
    epochs = args.epochs
    config_path.write_text(json.dumps({
        "seed": args.seed, "augmentation": not args.no_augmentation,
        "epochs": epochs, "batch_size": 64, "learning_rate": 0.001,
        "patience": patience, "train_size": train_size, "val_size": val_size,
        "torch": str(torch.__version__), "device": str(device),
    }, indent=2), encoding="utf-8")

    for epoch in range(epochs):
        model.train()#设置模型为训练模式

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)#将数据移动到设备上

            optimizer.zero_grad()#梯度清零

            outputs = model(images)#前向传播
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#反向传播
            optimizer.step()#更新权重

            train_loss += loss.item() * images.size(0)#累计训练损失
            predicted = outputs.argmax(dim=1)#获取预测结果
            train_total += labels.size(0)#累计训练样本数
            train_correct += (predicted == labels).sum().item()#累计正确预测数

        #计算整个指标
        train_loss /= train_total
        train_accuracy = train_correct / train_total

        model.eval()#设置模型为评估模式

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():#关闭梯度计算
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)#将数据移动到设备上

                outputs = model(images)#前向传播
                loss = criterion(outputs, labels)#计算损失

                val_loss += loss.item() * images.size(0)#累计验证损失
                predicted = outputs.argmax(dim=1)#获取预测结果
                val_total += labels.size(0)#累计验证样本数
                val_correct += (predicted == labels).sum().item()#累计正确预测数

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        #打印训练和验证指标
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 保存每轮指标；loss 已按样本数加权，最后一批较小也不会偏。
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
        })
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        #early stopping
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError("损失出现 NaN/Inf，本轮停止且不保存。")
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            wait = 0
            best_model_state = copy.deepcopy(model.state_dict())#保存最佳模型权重
            torch.save(best_model_state, model_path)
            print("保存最佳模型权重。")
        else:
            wait += 1
            print(f"验证损失没有改善，等待次数: {wait}/{patience}")
            if wait >= patience:
                print("早停触发，停止训练。")
                break

    #恢复最佳模型权重：保留你的 deepcopy 写法，防止最佳权重随训练更新。
    model.load_state_dict(best_model_state)

    #保存最佳模型权重
    print(f"最佳模型权重已保存到 {model_path}")


if __name__ == "__main__":
    main()
