import torch
import matplotlib.pyplot as plt

from network import CNN
from load_data import test_dataset


# =========================================================
# 1. 设备
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. 模型
# =========================================================

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "mnist/models/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# =========================================================
# 3. 取一张测试图片
# =========================================================

image, label = test_dataset[0]

x = image.unsqueeze(0).to(device)

print("真实标签：", label)
print("输入 shape：", x.shape)


# =========================================================
# 4. 前向传播到各个中间位置
# =========================================================

with torch.no_grad():

    # -----------------------------
    # 第一层
    # -----------------------------

    conv1 = model.conv1[0](x)

    relu1 = model.conv1[1](conv1)

    pool1 = model.conv1[2](relu1)


    # -----------------------------
    # 第二层
    # -----------------------------

    conv2 = model.conv2[0](pool1)

    relu2 = model.conv2[1](conv2)

    pool2 = model.conv2[2](relu2)


# =========================================================
# 5. 打印形状
# =========================================================

print("\n第一层：")
print("Conv1 :", conv1.shape)
print("ReLU1 :", relu1.shape)
print("Pool1 :", pool1.shape)

print("\n第二层：")
print("Conv2 :", conv2.shape)
print("ReLU2 :", relu2.shape)
print("Pool2 :", pool2.shape)


# =========================================================
# 6. 显示原图
# =========================================================

plt.figure(figsize=(3, 3))

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(f"Original - Label: {label}")

plt.axis("off")

plt.show()


# =========================================================
# 7. 第一层 16 张特征图
# =========================================================

feature_maps1 = relu1.squeeze(0).cpu()

plt.figure(figsize=(10, 10))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        feature_maps1[i],
        cmap="gray"
    )

    plt.title(f"Conv1-{i + 1}")

    plt.axis("off")

plt.tight_layout()

plt.show()


# =========================================================
# 8. 第二层 32 张特征图
# =========================================================

feature_maps2 = relu2.squeeze(0).cpu()

plt.figure(figsize=(12, 16))

for i in range(32):

    plt.subplot(8, 4, i + 1)

    plt.imshow(
        feature_maps2[i],
        cmap="gray"
    )

    plt.title(f"Conv2-{i + 1}")

    plt.axis("off")

plt.tight_layout()

plt.show()