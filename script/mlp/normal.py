import time
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ==========================================
# 1. 超参数
# ==========================================

torch.manual_seed(42)

num_epochs = 10
learning_rate = 0.5
batch_size = 256

dropout1 = 0.2
dropout2 = 0.5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. Fashion-MNIST 数据集
# ==========================================

transform = transforms.ToTensor()

full_train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 从 60000 张训练图片中划出验证集
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 3. MLP 模型
# ==========================================

net = nn.Sequential(
    nn.Flatten(),

    nn.Linear(784, 256),
    nn.ReLU(),

    nn.Linear(256, 256),
    nn.ReLU(),

    nn.Linear(256, 10)
)


# ==========================================
# 4. 权重初始化
# ==========================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(
            m.weight,
            mean=0,
            std=0.01
        )
        nn.init.zeros_(m.bias)


net.apply(init_weights)

net = net.to(device)


# ==========================================
# 5. 损失函数与优化器
# ==========================================

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=learning_rate
)


# ==========================================
# 6. 训练函数
# ==========================================

def train_one_epoch():

    net.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        # 前向传播（Dropout 启用）
        logits = net(X)

        loss = loss_func(logits, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        n = y.size(0)

        total_loss += loss.item() * n

        prediction = logits.argmax(dim=1)

        total_correct += (
            prediction == y
        ).sum().item()

        total_samples += n

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ==========================================
# 7. 评估函数
# ==========================================

def evaluate(loader):

    net.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            # 前向传播（Dropout 关闭）
            logits = net(X)

            loss = loss_func(logits, y)

            n = y.size(0)

            total_loss += loss.item() * n

            prediction = logits.argmax(dim=1)

            total_correct += (
                prediction == y
            ).sum().item()

            total_samples += n

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ==========================================
# 8. 保存训练历史
# ==========================================

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_loss = float("inf")
best_state = None


# ==========================================
# 9. 开始训练
# ==========================================

start_time = time.perf_counter()

for epoch in range(num_epochs):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            net.state_dict()
        )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


# ==========================================
# 10. 恢复最佳模型
# ==========================================

net.load_state_dict(best_state)

test_loss, test_acc = evaluate(test_loader)

if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.perf_counter()

print("\nFinal Results")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

print(
    f"Total Time: {end_time - start_time:.4f}s"
)


# ==========================================
# 11. 绘制损失曲线
# ==========================================

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Dropout MLP - Loss")

plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 12. 绘制准确率曲线
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dropout MLP - Accuracy")

plt.legend()
plt.grid(True)
plt.show()