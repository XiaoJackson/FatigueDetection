import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

# 设定随机种子以确保结果可复现
torch.manual_seed(1337)

# 检测是否有可用的 GPU，如果有，使用第一个 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载数据集
pickle_files = ['yawn_mouths.pickle']
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # 提示垃圾收集器释放内存

# 预处理数据
# 假设数据已经是适合的形状 (通道数, 高度, 宽度)
train_dataset = torch.tensor(train_dataset, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(device)
test_dataset = torch.tensor(test_dataset, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1).to(device)

train_data = TensorDataset(train_dataset, train_labels)
test_data = TensorDataset(test_dataset, test_labels)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# 定义模型
class mouthCNN(nn.Module):
    def __init__(self):
        super(mouthCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 新增的卷积层
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)  # 新增的卷积层
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 13 * 13, 512)  # 修改后的全连接层输入大小
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))  # 新增的卷积层
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

# 实例化模型并设置优化器和损失函数
model = mouthCNN().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
criterion = nn.BCELoss()

# 训练模型
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.squeeze(0)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'训练轮次 {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

for epoch in range(1, 21):
    train(epoch)

# 评估模型
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        target = target.squeeze(0)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = (output > 0.5).float()  # 将概率大于0.5视为类别1
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'\n测试集上的平均损失: {test_loss:.4f}, 准确率: {accuracy:.0f}%\n')

# 保存模型
torch.save(model.state_dict(), 'mouth_status_model.pth')