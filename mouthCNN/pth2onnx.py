import torch
import torch.onnx
import onnx
import torch.nn as nn

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

# 假设 model 是你的 PyTorch 模型，它已经被加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#加载嘴巴模型
mouth_model = mouthCNN()
mouth_model.load_state_dict(torch.load("mouth_status_model.pth"))
mouth_model = mouth_model.to('cpu')
mouth_model.eval()

# 创建一个相应的输入张量，注意它的形状应该符合你的模型输入，这里只是一个例子
dummy_input = torch.randn(1, 1, 60, 60)
dummy_input = dummy_input.to('cpu')

# 设定 ONNX 文件的保存路径
onnx_file_path = 'mouthmodel.onnx'

# 导出模型
torch.onnx.export(mouth_model, dummy_input, onnx_file_path, verbose=True, opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# 检查 ONNX 模型是否正确导出
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)
