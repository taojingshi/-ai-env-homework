# mnist_mlp.py  MNIST手写数字识别（MLP多层感知机）
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ---------------------- 超参数配置 ----------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {DEVICE}")

# ---------------------- 数据加载与预处理 ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 自动下载MNIST数据集
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------- 模型定义 ----------------------
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # 28×28图片展平成一维向量
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)          # 输出0~9共10个类别
        )

    def forward(self, x):
        return self.net(x)

model = MNISTModel().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------- 训练函数 ----------------------
def train_epoch():
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ---------------------- 测试函数，计算准确率 ----------------------
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            predict = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    accuracy = correct / total
    return accuracy

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    print("===== MNIST手写数字识别 开始训练 =====")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch()
        test_acc = test_model()
        print(f"Epoch {epoch:2d} | 训练损失: {train_loss:.4f} | 测试准确率: {test_acc:.4f}")
    end = time.time()
    print(f"\n训练结束，耗时 {end - start:.2f} 秒")
    final_acc = test_model()
    print(f"【最终测试集准确率】：{final_acc:.4f}")