import torch
import torch.onnx
import onnx
import torch.nn as nn

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
# 假设 model 是你的 PyTorch 模型，它已经被加载
# 加载眼睛状态模型
eye_model = EyeStatusCNN()
eye_model.load_state_dict(torch.load("eye_status_model.pth"))
eye_model = eye_model.to('cpu')
eye_model.eval()

# 创建一个相应的输入张量，注意它的形状应该符合你的模型输入，这里只是一个例子
dummy_input = torch.randn(1, 1, 24, 24)
dummy_input = dummy_input.to('cpu')

# 设定 ONNX 文件的保存路径
onnx_file_path = 'eyemodel.onnx'

# 导出模型
torch.onnx.export(eye_model, dummy_input, onnx_file_path, verbose=True, opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# 检查 ONNX 模型是否正确导出
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)