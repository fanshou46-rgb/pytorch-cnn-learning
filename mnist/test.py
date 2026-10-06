"""Final test-set evaluation for the MNIST CNN."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from load_data import get_dataset
from network import CNN

from sklearn.metrics import classification_report, confusion_matrix


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here / "models" / "best_model_augmented.pth")
    parser.add_argument("--no-plot", action="store_true", help="不弹出绘图窗口")
    parser.add_argument("--save-dir", type=Path, help="保存指标、逐张预测和图像")
    args = parser.parse_args()
    if args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print(f"模型文件：{args.model}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备：", device)
    test_dataset = get_dataset(train=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model = CNN().to(device)

    model.load_state_dict(
        torch.load(
            args.model,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    test_total = 0
    test_correct = 0
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    prediction_rows = []
    # 用来保存所有真实标签和预测标签
    all_labels = []
    all_predictions = []
    wrong_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            test_loss += criterion(outputs, labels).item() * labels.size(0)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(dim=1)

            wrong_mask = predicted != labels
            wrong_indices = torch.where(wrong_mask)[0]

            for i in wrong_indices.tolist():

                wrong_samples.append({
                    "image": images[i].cpu(),
                    "index": test_total + i,
                    "true": labels[i].item(),
                    "pred": predicted[i].item(),
                    "confidence": confidence[i].item(),
                })

            for i in range(labels.size(0)):
                prediction_rows.append({
                    "index": test_total + i, "true": labels[i].item(),
                    "pred": predicted[i].item(), "confidence": confidence[i].item(),
                })
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_labels.extend(
                labels.cpu().numpy()
            )#保存真实标签，extend是把元素拆开加进去

            all_predictions.extend(
                predicted.cpu().numpy()
            )#保存预测标签

    # 收集所有错图，再按置信度排序，最后只展示前 9 张。
    wrong_samples.sort(key=lambda sample: sample["confidence"], reverse=True)
    print("错分数量：", len(wrong_samples))
    print(f"Test Loss: {test_loss / test_total:.4f}")

    test_accuracy = (
        test_correct / test_total
    )

    print("测试样本数量：", test_total)
    print("预测正确数量：", test_correct)
    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # =========================================================
    # 7. 混淆矩阵
    # =========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10)),  # 即使有类别没出现，也固定为 10×10
    )

    print("\n混淆矩阵：")
    print(cm)


    # 分类报告补充每一类的 Precision / Recall / F1。
    report = classification_report(all_labels, all_predictions, labels=list(range(10)),
                                   output_dict=True, zero_division=0)
    print(classification_report(all_labels, all_predictions, labels=list(range(10)),
                                digits=4, zero_division=0))
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "samples": test_total, "correct": test_correct, "errors": len(wrong_samples),
            "accuracy": test_accuracy, "loss": test_loss / test_total,
            "classification_report": report, "confusion_matrix": cm.tolist(),
            "model": str(args.model.resolve()),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        }
        (args.save_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8",
        )
        with (args.save_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)

    # no-plot 不弹窗；指定 save-dir 时仍生成并保存图片。
    if not args.no_plot or args.save_dir:
        # 混淆矩阵
        plt.figure(figsize=(8, 8))#创建画布

        plt.imshow(cm)#渲染热力图

        plt.title("MNIST Confusion Matrix")

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.xticks(range(10))#刻度
        plt.yticks(range(10))

        plt.colorbar()

        # 在每个格子里写数字
        for i in range(10):
            for j in range(10):

                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",#居中
                    va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black",
                )

        plt.tight_layout()#调整间距
        if args.save_dir:
            plt.savefig(args.save_dir / "confusion_matrix.png", dpi=160)

        if wrong_samples:
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))

            for ax in axes.flat:
                ax.axis("off")

            for ax, sample in zip(axes.flat, wrong_samples[:9]):

                ax.imshow(
                    sample["image"].squeeze(0),
                    cmap="gray"
                )

                ax.set_title(
                    f'True: {sample["true"]} | Pred: {sample["pred"]}\n'
                    f'Confidence: {sample["confidence"] * 100:.1f}%'
                )

            fig.tight_layout()
            if args.save_dir:
                fig.savefig(args.save_dir / "wrong_samples.png", dpi=160)

        if not args.no_plot:
            plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
