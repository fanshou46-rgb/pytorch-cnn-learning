"""MNIST dataset and DataLoader setup.

This file will be implemented step by step during the CNN learning project.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])#图像预处理，将图像转换为张量

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)#下载训练集数据

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)#下载测试集数据

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,num_workers=0)#训练集数据加载器

test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False,num_workers=0)#测试集数据加载器

if __name__ == "__main__":#
    print("训练集数量：",len(train_dataset))
    print("测试集数量：",len(test_dataset))

    image, label = next(iter(train_loader))

    print("image shape = ", image.shape)
    print("label shape = ", label.shape)

    print("第一张图片的标签 = ", label[0].item())
    print("第一张图片的shape = ", image[0].shape)

    