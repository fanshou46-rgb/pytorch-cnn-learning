"""Single-image prediction with the trained MNIST CNN."""
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from network import CNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "mnist/models/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

#读取图片
image = Image.open(
    "mnist/images/test.png"
)


# 转为灰度图
image = image.convert("L")

# 调整为 28 × 28

image = image.resize(
    (28, 28)
)

# 反色（如果是白底黑字）
#image = ImageOps.invert(image)



# 换为 Tensor


transform = transforms.ToTensor()

image = transform(image)



# 增加 batch 维度
image = image.unsqueeze(0)

image = image.to(device)

# 预测
with torch.no_grad():

    output = model(image)

    predicted = output.argmax(dim=1)


# 输出结果
print("模型输出：")
print(output)

print(
    "预测数字：",
    predicted.item()
)