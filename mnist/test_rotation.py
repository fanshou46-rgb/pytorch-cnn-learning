"""固定旋转测试：用同一批图片比较基线模型和增强模型。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from load_data import get_dataset
from network import CNN


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=here / "models/baseline_seed42.pth")
    parser.add_argument("--augmented", type=Path, default=here / "models/augmented_seed42.pth")
    parser.add_argument("--save-dir", type=Path,
                        default=here.parent / "docs/results/rotation_seed42")
    args = parser.parse_args()
    csv_path = args.save_dir / "metrics.csv"
    config_path = args.save_dir / "config.json"
    if csv_path.exists() or config_path.exists():
        parser.error("结果文件已存在，请用 --save-dir 指定新的目录。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device, flush=True)
    model_paths = {"baseline": args.baseline, "augmented": args.augmented}
    models = {}
    for name, path in model_paths.items():
        model = CNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models[name] = model

    # 先取官方测试集，随后指定确定性的变换；不使用训练集的随机增强。
    test_dataset = get_dataset(train=False)
    rows = []
    angles = (-10, 0, 10)
    for angle in angles:
        # 对 PIL 原图旋转，再转 Tensor，与训练增强的处理顺序一致。
        # 正角度逆时针，负角度顺时针；最近邻插值、黑色填充，画布仍是 28×28。
        test_dataset.transform = transforms.Compose([
            transforms.Lambda(lambda image, angle=angle: TF.rotate(
                image, angle, interpolation=transforms.InterpolationMode.NEAREST,
                expand=False, fill=0,
            )),
            transforms.ToTensor(),
        ])
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        total = baseline_correct = augmented_correct = 0
        corrected = regressed = both_wrong = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # 同一个 images 分别交给两个模型，避免输入存在差异。
                baseline_pred = models["baseline"](images).argmax(dim=1)
                augmented_pred = models["augmented"](images).argmax(dim=1)
                baseline_ok = baseline_pred == labels
                augmented_ok = augmented_pred == labels
                total += labels.size(0)
                baseline_correct += baseline_ok.sum().item()
                augmented_correct += augmented_ok.sum().item()
                corrected += (~baseline_ok & augmented_ok).sum().item()
                regressed += (baseline_ok & ~augmented_ok).sum().item()
                both_wrong += (~baseline_ok & ~augmented_ok).sum().item()

        row = {
            "angle": angle, "samples": total,
            "baseline_correct": baseline_correct,
            "baseline_accuracy": baseline_correct / total,
            "baseline_errors": total - baseline_correct,
            "augmented_correct": augmented_correct,
            "augmented_accuracy": augmented_correct / total,
            "augmented_errors": total - augmented_correct,
            "corrected": corrected, "regressed": regressed, "both_wrong": both_wrong,
        }
        rows.append(row)
        print(f"角度 {angle:+d}° | 基线 {baseline_correct / total:.2%} "
              f"({total - baseline_correct} 张错) | 增强 {augmented_correct / total:.2%} "
              f"({total - augmented_correct} 张错)", flush=True)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "angles": angles, "dataset": "MNIST official test set",
        "interpolation": "NEAREST", "fill": 0, "expand": False,
        "device": str(device), "torch_version": torch.__version__,
        "models": {
            name: {"path": str(path.resolve()),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in model_paths.items()
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"结果已保存：{csv_path}")


if __name__ == "__main__":
    main()
