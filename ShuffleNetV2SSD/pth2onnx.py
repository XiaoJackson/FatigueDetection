import torch
import torch.onnx
import onnx
from ShuffleNetV2SSD import ShuffleNetV2SSD

# 假设 model 是你的 PyTorch 模型，它已经被加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShuffleNetV2SSD(n_classes=4)
model.load_state_dict(torch.load('featuredetection.pth'))
model = model.to('cpu')
model.eval()

# 创建一个相应的输入张量，注意它的形状应该符合你的模型输入，这里只是一个例子
dummy_input = torch.randn(1, 3, 300, 300)
dummy_input = dummy_input.to('cpu')

# 设定 ONNX 文件的保存路径
onnx_file_path = 'ssd2feature.onnx'

# 导出模型
torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True, opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# 检查 ONNX 模型是否正确导出
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)