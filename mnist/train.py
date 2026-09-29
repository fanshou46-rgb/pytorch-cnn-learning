"""Training and validation loop for the MNIST CNN.

Early stopping and model checkpointing will be added here.
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from load_data import train_dataset
from network import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备: ", device)#选择设备，优先使用GPU，如果没有GPU则使用CPU

#划分训练集和验证集
train_size = 50000
val_size = 10000

train_data,val_data = random_split(train_dataset,[train_size,val_size])#划分训练集和验证集

train_loader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=0)#训练集数据加载器
val_loader = DataLoader(val_data, batch_size=64, shuffle=False,num_workers=0)#验证集数据加载器

model = CNN().to(device)#实例化CNN模型并将其移动到设备上

criterion = nn.CrossEntropyLoss()#定义损失函数

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#定义优化器

#early stopping参数
best_val_loss = float('inf')#初始化最佳验证损失为无穷大
patience = 3
wait =0
best_model_state = None#保存最佳模型权重

#训练参数
epoches = 20
for epoch in range(epoches):
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
    print(f"Epoch [{epoch+1}/{epoches}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    #early stopping
    if val_loss<best_val_loss:

        best_val_loss = val_loss
        wait = 0
        best_model_state = copy.deepcopy(model.state_dict())#保存最佳模型权重
        print("保存最佳模型权重。")
    else:
        wait += 1
        print(f"验证损失没有改善，等待次数: {wait}/{patience}")
        if wait >= patience:
            print("早停触发，停止训练。")
            break

    #恢复最佳模型权重
model.load_state_dict(best_model_state)

#保存最佳模型权重
torch.save(model.state_dict(), "mnist/models/best_model.pth")
print("最佳模型权重已保存到 mnist/models/best_model.pth")
