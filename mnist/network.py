"""CNN model definition for MNIST.

This file will contain the convolutional neural network architecture.
"""
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),#卷积层1
            nn.ReLU(),#激活函数1
            nn.MaxPool2d(kernel_size=2)#池化层1
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),#卷积层2
            nn.ReLU(),#激活函数2  
            nn.MaxPool2d(kernel_size=2)#池化层2
        )

        self.fc = nn.Linear(32 * 7 * 7 , 10)#全连接层
    def forward(self,x):
        x=self.conv1(x)
        x= self.conv2(x)
        x=x.view(x.size(0), -1)#展平
        x=self.fc(x)
        return x
if __name__ == "__main__":
    model = CNN()
    x=torch.randn(64, 1, 28, 28)#正分布随机数
    output = model(x)
    print("输入 shape = ", x.shape)
    print("输出 shape = ", output.shape)
    