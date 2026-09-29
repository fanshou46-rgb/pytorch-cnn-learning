import torch
import matplotlib.pyplot as plt

from network import CNN


# =========================================================
# 1. 设备
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. 创建模型
# =========================================================

model = CNN().to(device)


# =========================================================
# 3. 加载训练好的参数
# =========================================================

model.load_state_dict(
    torch.load(
        "mnist/models/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# =========================================================
# 4. 取出第一层卷积核
# =========================================================

kernels = model.conv1[0].weight.detach().cpu()

print("第一层卷积核 shape：", kernels.shape)
# =========================================================
# 5. 移到 CPU
# =========================================================

kernels = kernels.cpu()


# =========================================================
# 6. 显示 16 个卷积核
# =========================================================

plt.figure(figsize=(8, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    # kernels[i] 的 shape 是 [1, 3, 3]
    kernel = kernels[i].squeeze(0)

    plt.imshow(
        kernel,
        cmap="gray"
    )

    plt.title(f"Kernel {i + 1}")

    plt.axis("off")


plt.tight_layout()
plt.show()