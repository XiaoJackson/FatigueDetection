import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn

# 定义你的模型结构（根据你实际使用的模型进行调整）
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

# 加载模型
def load_model(model_path):
    model = EyeStatusCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 图片处理转换
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),  # 确保输入为单通道灰度
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图标准化
])

# 对图片文件夹进行遍历，并进行预测
def predict_images(folder_path, model):
    images = os.listdir(folder_path)
    open_count = 0
    closed_count = 0
    total = 0

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('L')  # 尝试打开图像
        except IOError:
            print(f"Cannot open image: {img_path}")
            continue  # 跳过无法打开的文件
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # 增加批次维度

        with torch.no_grad():
            output = model(img_tensor)
            # 由于模型输出一个sigmoid概率，我们用0.5作为阈值来决定类别
            predicted = (output > 0.5).item()  # True if 'Open', False if 'Closed'

        if predicted:
            status = 'Open'
            open_count += 1
        else:
            status = 'Closed'
            closed_count += 1
        total += 1
        print(f'{img_name}: Eye is {status}')

    if total > 0:
        print("\nStatistics:")
        print(f"Total images: {total}")
        print(f"Open Eyes: {open_count} ({open_count / total * 100:.2f}%)")
        print(f"Closed Eyes: {closed_count} ({closed_count / total * 100:.2f}%)")
    else:
        print("No valid images were found to predict.")

# 主函数
if __name__ == '__main__':
    model_path = 'eye_status_model.pth'  # 模型文件路径
    images_folder_path = r'F:\aria2\dataset\dataset_B_Eye_Images\closedRightEyes'  # 图片文件夹路径

    model = load_model(model_path)
    predict_images(images_folder_path, model)
