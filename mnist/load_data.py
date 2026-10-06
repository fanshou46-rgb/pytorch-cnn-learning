"""MNIST 数据读取：只在调用 get_dataset() 时加载数据。"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 沿用根目录已有 data/ 缓存，避免随运行目录改变下载位置。
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # 轻微旋转，不改变数字类别
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 轻微平移
    transforms.ToTensor(),
])
test_transform = transforms.ToTensor()


def get_dataset(train=True, augment=False):
    """训练图片可增强；验证和测试图片只转为 Tensor。"""
    if augment and not train:
        raise ValueError("测试集不能使用随机增强。")
    return datasets.MNIST(
        root=str(DATA_DIR), train=train, download=True,
        transform=train_transform if augment else test_transform,
    )


if __name__ == "__main__":
    # 单独运行这个文件时，才下载数据并检查形状。
    train_dataset = get_dataset(train=True, augment=True)
    test_dataset = get_dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("官方训练集数量：", len(train_dataset))
    print("官方测试集数量：", len(test_dataset))
    images, labels = next(iter(train_loader))
    print("images shape =", images.shape)
    print("labels shape =", labels.shape)
