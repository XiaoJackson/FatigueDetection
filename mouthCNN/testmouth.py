import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# 假设你的模型类和预训练模型已经定义好了
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


# 图片处理转换
transform = transforms.Compose([
    transforms.Resize((60, 60)),  # 调整图像大小以匹配模型输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图标准化
])

# 加载模型
def load_model(model_path):
    model = mouthCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 对图片文件夹进行遍历，并进行预测
def predict_images(folder_path, model):
    images = os.listdir(folder_path)
    open_count = 0
    closed_count = 0
    total = 0

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('L')  # 打开并转换为灰度图像
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0)  # 增加批次维度

            with torch.no_grad():
                output = model(img_tensor)
                predicted = (output > 0.9).item()
            print(predicted)

            if predicted:
                status = 'Yawn'
                open_count += 1
            else:
                status = 'No Yawn'
                closed_count += 1
            total += 1
            print(f'{img_name}: Mouth is {status}')

        except IOError:
            print(f"Cannot open image: {img_path}")
            continue  # 跳过无法打开的文件

    if total > 0:
        print("\nStatistics:")
        print(f"Total images: {total}")
        print(f"Yawn: {open_count} ({open_count / total * 100:.2f}%)")
        print(f"No Yawn: {closed_count} ({closed_count / total * 100:.2f}%)")
    else:
        print("No valid images were found to predict.")

# 主函数
if __name__ == '__main__':
    model_path = 'mouth_status_model.pth'  # 模型文件路径
    images_folder_path = r'F:\aria2\dataset\archive\yawn'  # 图片文件夹路径

    model = load_model(model_path)
    predict_images(images_folder_path, model)