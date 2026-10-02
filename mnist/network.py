"""MNIST CNN：两组卷积、ReLU、池化，最后输出 10 个 logits。"""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)  # [N, 1, 28, 28] → [N, 16, 14, 14]
        x = self.conv2(x)  # → [N, 32, 7, 7]
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)  # CrossEntropyLoss 直接接收 logits。


if __name__ == "__main__":
    model = CNN()
    x = torch.randn(64, 1, 28, 28)
    print("输入 shape =", x.shape)
    print("输出 shape =", model(x).shape)
