import argparse
from pathlib import Path

from load_data import get_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not 0 <= args.index < 60000:
        parser.error("index 超出训练集范围。")
    train_dataset = get_dataset(train=True, augment=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for ax in axes.flat:

        image, label = train_dataset[args.index]

        ax.imshow(
            image.squeeze(),
            cmap="gray"
        )

        ax.set_title(f"label = {label}")
        ax.axis("off")

    plt.tight_layout()
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_dir / "augmentation.png", dpi=160)
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
