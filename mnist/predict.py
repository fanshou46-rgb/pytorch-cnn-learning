"""Single-image prediction with the trained MNIST CNN."""
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from network import CNN


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here / "models" / "best_model_augmented.pth")
    parser.add_argument("--image", type=Path, default=here / "images" / "test.png")
    parser.add_argument("--invert", action="store_true", help="白底黑字图片需要反色")
    args = parser.parse_args()
    print(f"模型文件：{args.model}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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

    #读取图片
    with Image.open(args.image) as source:
        image = source.convert("L")

    # 转为灰度图
    image = image.convert("L")

    # 调整为 28 × 28

    image = image.resize(
        (28, 28)
    )

    # 反色（如果是白底黑字）
    if args.invert:
        image = ImageOps.invert(image)


    # 换为 Tensor

    transform = transforms.ToTensor()

    image = transform(image)


    # 增加 batch 维度
    image = image.unsqueeze(0)

    image = image.to(device)

    # 预测
    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    # 输出结果
    print("模型输出：")
    print(output)

    print(
        "预测数字：",
        predicted.item()
    )
    print(f"Softmax 置信度：{confidence.item() * 100:.2f}%")
    top_probabilities, top_digits = probabilities[0].topk(3)
    print("概率最高的 3 个数字：")
    for digit, probability in zip(top_digits.tolist(), top_probabilities.tolist()):
        print(f"  {digit}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
