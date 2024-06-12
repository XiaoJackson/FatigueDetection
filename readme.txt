依赖环境：
Python >= 3.7.2
opencv-python >= 4.9.0.80
opencv-contrib-python >= 4.9.0.80
torch >= 1.13.0+cu117
torchvision >= 0.14.0+cu117
onnx >= 1.8.1
onnxruntime >= 1.8.1
MNN >= 2.8.1
使用说明：
运行根目录下FatigueDetection.py即可运行本设计。
文件目录及说明如下：
/root                                                               #根目录
│
├── /eyeCNN                                                # 眼睛状态判断模型文件夹
│   ├── closed_eyes.pickle                             # 闭眼数据集数据
│   ├── dataloader.py                                   # 数据加载
│   ├── eye_status_model.pth                       # 训练好的眼睛状态判断模型(Pytorch版本)
│   ├── eyemodel.onnx                                # 训练好的眼睛状态判断模型(Onnx版本)
│   ├── open_eyes.pickle                              # 睁眼数据集数据
│   ├── pth2onnx.py                                    # pth模型转onnx模型
│   ├── train.py                                            # 训练代码
│   └── testeye.py                                        # 测试模型代码
│
├── /mouthCNN                                          # 嘴巴状态判断模型文件夹
│   ├── dataloader.py                                  # 数据加载
│   ├── model.py                                         # 嘴巴模型所用卷积神经网络
│   ├── mouth_status_model.pth                 # 训练好的嘴巴状态判断模型(Pytorch版本)
│   ├── mouthmodel.onnx                           # 训练好的嘴巴状态判断模型(Onnx版本)
│   ├── pth2onnx.py                                    # pth模型转onnx模型
│   ├── testmouth.py                                   # 测试模型代码
│   ├── train.py                                            # 训练代码
│   └── yawn_mouths.pickle                        # 嘴巴数据集数据
│
├── /ShuffleNetV2SSD                                # ShuffleNetV2SSD模型文件夹
│   ├── /testSSD                                          # 测试ShuffleNetV2SSD模型
│   │   ├── box_utils_numpy.py                   # 人脸检测器依赖代码
│   │   ├── RFB-320.mnn                             # 人脸检测器模型
│   │   ├── RFB-320-quant-KL-5792.mnn    # 人脸检测器模型(int8量化)
│   │   ├── testpthver.py                              # ShuffleNetV2SSD模型测试代码
│   │   └── voc-model-labels.txt                  # 人脸检测器依赖类别表
│   ├── 300wboxes1.xml                             # 训练所用数据集的标注信息文件(开发机)
│   ├── 300wboxes2.xml                            # 训练所用数据集的标注信息文件(服务器)
│   ├── checkpoint_ssd300.pth.tar              # 训练好的ShuffleNetV2疲劳特征提取模型(Pyroch检查点模型版本)
│   ├── Dataloader.py                                 # 数据加载
│   ├── featuredetection.pth                       # 训练好的ShuffleNetV2疲劳特征提取模型(Pyroch状态字典版本)
│   ├── pth2onnx.py                                   # pth模型转onnx模型
│   ├── ShuffleNetV2SSD.py                       # ShuffleNetV2SSD代码
│   ├── ssd2feature.onnx                            # 训练好的ShuffleNetV2疲劳特征提取模型(Onnx版本)
│   ├── train.py                                           # 训练代码
│   └── utils.py                                            # ShuffleNetV2SSD实现依赖代码
│
├── box_utils_numpy.py                             # 人脸检测器依赖代码
├── eyemodel.onnx                                    # 训练好的眼睛状态判断模型(Onnx版本)
├── FatigueDetection.py                             # 疲劳驾驶检测实现代码
├── mouthmodel.onnx                               # 训练好的嘴巴状态判断模型(Onnx版本)
├── RFB-320.mnn                                       # 人脸检测器模型
├── RFB-320-quant-KL-5792.mnn             # 人脸检测器模型(int8量化)
├── ShuffleNetV2SSD.py                            # ShuffleNetV2SSD代码
├── ssd2feature.onnx                                 # 训练好的ShuffleNetV2疲劳特征提取模型(Onnx版本)
├── utils.py                                                 # ShuffleNetV2SSD实现依赖代码
├── voc-model-labels.txt                            # 人脸检测器依赖类别表
└── 环境配置及使用说明.txt                          # 说明及配置文件

