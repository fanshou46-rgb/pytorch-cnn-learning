"""Final test-set evaluation for the MNIST CNN."""
import torch
from torch.utils.data import DataLoader
from load_data import test_dataset
from network import CNN


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
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

test_total = 0
test_correct = 0
# 用来保存所有真实标签和预测标签
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)#返回每行最大值的索引，即预测的类别

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_labels.extend(
            labels.cpu().numpy()
        )#保存真实标签，extend是把元素拆开加进去

        all_predictions.extend(
            predicted.cpu().numpy()
        )#保存预测标签


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
    all_predictions
)

print("\n混淆矩阵：")
print(cm)



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
            va="center"
        )


plt.tight_layout()#调整间距

plt.show()