import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from torchvision import transforms

# 设置随机种子以确保结果可复现
torch.manual_seed(1337)


# 加载和预处理数据
def load_data(pickle_files):
    datasets = []
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            datasets.append((save['train_dataset'], save['train_labels'],
                             save['test_dataset'], save['test_labels']))
            del save  # 帮助垃圾回收释放内存
    return datasets


def preprocess_data(datasets):
    train_data = np.vstack([data[0] for data in datasets])
    train_labels = np.vstack([data[1] for data in datasets])
    test_data = np.vstack([data[2] for data in datasets])
    test_labels = np.vstack([data[3] for data in datasets])

    # 将数据转换为 PyTorch 张量
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # 确保数据形状为 [batch, channel, height, width]，这里 channel 应该是 1
    # 有些情况下，原始数据可能是 [batch, height, width, channel]
    # 如果是这种情况，需要调整为 [batch, channel, height, width]
    train_data = train_data.reshape(train_data.shape[0], 1, 24, 24)
    test_data = test_data.reshape(test_data.shape[0], 1, 24, 24)

    # 创建张量数据集
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    return train_dataset, test_dataset


pickle_files = ['open_eyes.pickle', 'closed_eyes.pickle']
datasets = load_data(pickle_files)
train_dataset, test_dataset = preprocess_data(datasets)


# 定义模型
class EyeStatusCNN(nn.Module):
    def __init__(self):
        super(EyeStatusCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(24, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


# 训练模型
def train_model(model, train_dataset, test_dataset, batch_size=30, epochs=20):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    for epoch in range(epochs):
        model.train()
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, targets in test_loader:
                outputs = model(data)
                predicted = (outputs.data > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets.unsqueeze(1)).sum().item()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {100 * correct / total}%')


model = EyeStatusCNN()
train_model(model, train_dataset, test_dataset)

# 保存模型权重
torch.save(model.state_dict(), 'eye_status_model.pth')
