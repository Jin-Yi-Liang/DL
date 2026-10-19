import torch
import torchvision
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# =========================================================
# 1. 基础配置
# =========================================================

BATCH_SIZE = 256
NUM_EPOCHS = 5
LEARNING_RATE = 0.01

NUM_INPUTS = 28 * 28
NUM_CLASSES = 10

# 设置随机种子，尽可能保证实验可复现
torch.manual_seed(42)

# 自动选择 CPU 或 GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前训练设备:", device)


# =========================================================
# 2. 加载 Fashion-MNIST 数据集
# =========================================================

# 将图片转换为 Tensor，并把像素值缩放到 [0, 1]
transform = transforms.ToTensor()

# 训练集
train_dataset = torchvision.datasets.FashionMNIST(
    root="../data",
    train=True,
    transform=transform,
    download=True
)

# 测试集
test_dataset = torchvision.datasets.FashionMNIST(
    root="../data",
    train=False,
    transform=transform,
    download=True
)

print("训练集样本数量:", len(train_dataset))
print("测试集样本数量:", len(test_dataset))


# =========================================================
# 3. 创建 DataLoader
# =========================================================

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("训练集 Batch 数量:", len(train_loader))
print("测试集 Batch 数量:", len(test_loader))

# 查看 Batch 的 Tensor 形状
X, y = next(iter(train_loader))

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# 4. 构建 Softmax 回归模型
# =========================================================

model = nn.Sequential(

    # (B, 1, 28, 28) -> (B, 784)
    nn.Flatten(),

    # (B, 784) -> (B, 10)
    nn.Linear(NUM_INPUTS, NUM_CLASSES)
)

# 初始化权重
nn.init.normal_(
    model[1].weight,
    mean=0.0,
    std=0.01
)

# 初始化偏置
nn.init.zeros_(
    model[1].bias
)

# 将模型移动到 CPU 或 GPU
model = model.to(device)

print("\n模型结构:")
print(model)


# =========================================================
# 5. 定义损失函数
# =========================================================

# CrossEntropyLoss 内部完成 LogSoftmax + NLLLoss
# 模型直接输出 Logits，不需要手动调用 Softmax
criterion = nn.CrossEntropyLoss()


# =========================================================
# 6. 定义 SGD 优化器
# =========================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 7. 训练一个 Epoch
# =========================================================

def train_one_epoch():

    # 切换到训练模式
    model.train()

    # 用于统计整个 Epoch 的指标
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 遍历当前 Epoch 中的全部 Batch
    for X, y in train_loader:

        # 将当前 Batch 移动到训练设备
        X = X.to(device)
        y = y.to(device)

        # -------------------------------
        # Step 1：前向传播
        # -------------------------------

        logits = model(X)

        # logits shape: (B, 10)

        # -------------------------------
        # Step 2：计算交叉熵损失
        # -------------------------------

        loss = criterion(logits, y)

        # loss 是当前 Batch 的平均交叉熵

        # -------------------------------
        # Step 3：清空旧梯度
        # -------------------------------

        optimizer.zero_grad()

        # -------------------------------
        # Step 4：反向传播
        # -------------------------------

        loss.backward()

        # -------------------------------
        # Step 5：SGD 更新参数
        # -------------------------------

        optimizer.step()

        # -------------------------------
        # Step 6：统计当前 Batch 指标
        # -------------------------------

        batch_size = y.size(0)

        # 当前 Batch 的损失总和
        total_loss += loss.item() * batch_size

        # 得到每个样本的预测类别
        predictions = logits.argmax(dim=1)

        # 统计预测正确数量
        correct = (predictions == y).sum().item()

        total_correct += correct

        # 累计样本数量
        total_samples += batch_size

    # -----------------------------------
    # 计算整个 Epoch 的指标
    # -----------------------------------

    epoch_loss = total_loss / total_samples

    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# =========================================================
# 8. 测试模型
# =========================================================

def evaluate():

    # 切换到评估模式
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 测试阶段不需要计算梯度
    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            # 前向传播
            logits = model(X)

            # 计算交叉熵
            loss = criterion(logits, y)

            batch_size = y.size(0)

            # 累加损失
            total_loss += loss.item() * batch_size

            # 预测类别
            predictions = logits.argmax(dim=1)

            # 统计正确数量
            total_correct += (
                predictions == y
            ).sum().item()

            total_samples += batch_size

    # 测试集平均损失
    test_loss = total_loss / total_samples

    # 测试集准确率
    test_accuracy = total_correct / total_samples

    return test_loss, test_accuracy


# =========================================================
# 9. 完整训练循环
# =========================================================

# 记录训练历史
history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": []
}

print("\n========== 开始训练 ==========\n")

for epoch in range(NUM_EPOCHS):

    # 训练一个 Epoch
    train_loss, train_accuracy = train_one_epoch()

    # 在测试集上评估
    test_loss, test_accuracy = evaluate()

    # 保存指标
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["test_loss"].append(test_loss)
    history["test_accuracy"].append(test_accuracy)

    # 输出当前 Epoch 的训练结果
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy * 100:.2f}%"
    )

print("\n========== 训练完成 ==========\n")


# =========================================================
# 10. 绘制损失曲线
# =========================================================

epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    history["test_loss"],
    marker="s",
    label="Test Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Fashion-MNIST - Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "fashion_mnist_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 11. 绘制准确率曲线
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["test_accuracy"],
    marker="s",
    label="Test Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fashion-MNIST - Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "fashion_mnist_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 12. 保存模型参数
# =========================================================

torch.save(
    model.state_dict(),
    "fashion_mnist_model.pth"
)

print("模型参数已保存：fashion_mnist_model.pth")


# =========================================================
# 13. 使用训练完成的模型进行预测
# =========================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 获取测试集中的一个 Batch
X, y = next(iter(test_loader))

model.eval()

with torch.no_grad():

    # 模型输出 Logits
    logits = model(X.to(device))

    # 得到预测类别
    predictions = logits.argmax(dim=1).cpu()

    # 转换为概率，便于查看预测置信度
    probabilities = torch.softmax(
        logits,
        dim=1
    ).cpu()

# 显示 6 张测试图片
fig, axes = plt.subplots(
    1,
    6,
    figsize=(15, 3.5)
)

for i in range(6):

    image = X[i].squeeze(0)

    true_label = class_names[int(y[i])]

    predicted_index = int(predictions[i])
    predicted_label = class_names[predicted_index]

    confidence = float(
        probabilities[i, predicted_index]
    )

    axes[i].imshow(
        image,
        cmap="gray"
    )

    axes[i].set_title(
        f"True: {true_label}\n"
        f"Pred: {predicted_label}\n"
        f"Prob: {confidence:.2f}",
        fontsize=9
    )

    axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    "fashion_mnist_predictions.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("预测图片已保存：fashion_mnist_predictions.png")
print("损失曲线已保存：fashion_mnist_loss.png")
print("准确率曲线已保存：fashion_mnist_accuracy.png")