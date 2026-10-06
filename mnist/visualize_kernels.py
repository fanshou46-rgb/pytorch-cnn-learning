import argparse
from pathlib import Path
import torch

from network import CNN


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here / "models" / "best_model_augmented.pth")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
            args.model,
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

    # 上面已经移到 CPU，这里不需要重复调用。

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
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=160)
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
