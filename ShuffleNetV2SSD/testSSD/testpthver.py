from __future__ import print_function
import cv2
import torch
from torchvision import transforms
import numpy as np
from utils import *  # 确保这里包含了所有必需的映射和函数
import box_utils_numpy as box_utils
import time
from ShuffleNetV2SSD import *
import onnxruntime
from math import ceil
import MNN
from PIL import Image

def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [ceil(size / stride) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = ShuffleNetV2SSD(n_classes=4)
model.load_state_dict(torch.load('../featuredetection.pth'))
model = model.to(device)
model.eval()

# 人脸检测器默认参数设置
model_path = "RFB-320.mnn"
input_size = "320,240"
results_path = "results"

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]


# 视频捕获初始化
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Default resolution: {int(width)}x{int(height)}")
threshold = 0.7
fps = 0.0
last_time = time.time()
# 转换操作定义
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_eye = transforms.Resize((24, 24))  # 调整眼睛图像大小到24x24
input_size_split = [int(v.strip()) for v in input_size.split(",")]
priors = define_img_size(input_size_split)
interpreter = MNN.Interpreter(model_path)
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)
while True:
    ret, frame = cap.read()
    if frame is None:
        print("No image from camera.")
        break

    image_ori = frame
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size_split[0], input_size_split[1]))  # 使用元组提供正确的尺寸
    image = (image - image_mean) / image_std
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)

    tmp_input = MNN.Tensor((1, 3, input_size_split[1], input_size_split[0]), MNN.Halide_Type_Float, image,
                           MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)

    start_time = time.time()
    interpreter.runSession(session)
    scores = interpreter.getSessionOutput(session, "scores").getData()
    boxes = interpreter.getSessionOutput(session, "boxes").getData()
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)

    boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = box_utils.center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, threshold)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        face_roi = frame[box[1]:box[3], box[0]:box[2]]
        face_resized = cv2.resize(face_roi, (300, 300))

        # 转换为PIL图像以应用Torchvision变换
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

        # 应用转换
        face_tensor = to_tensor(resize(face_pil)).unsqueeze(0)
        face_tensor = normalize(face_tensor).to(device)


        # 模型推理
        with torch.no_grad():
            predicted_locs, predicted_scores = model(face_tensor)

            # 从模型输出解析检测框和标签
            det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.4,
                                                                     max_overlap=0.6, top_k=200)

            det_boxes = det_boxes[0].to('cpu')
            det_labels = det_labels[0].to('cpu')
            det_scores = det_scores[0].to('cpu')

            # 调整尺寸到原图
            original_dims = torch.FloatTensor(
                [face_roi.shape[1], face_roi.shape[0], face_roi.shape[1], face_roi.shape[0]]).unsqueeze(0)
            det_boxes = det_boxes * original_dims

            # 选择每个框中置信度最高的标签
            unique_labels = det_labels.unique(sorted=True)
            best_label_for_box = []
            for label in unique_labels:
                mask = (det_labels == label)
                if mask.any():
                    best_score, best_idx = det_scores[mask].max(0)
                    best_label_for_box.append((det_boxes[mask][best_idx], label, best_score))

            # 绘制检测结果
            for box, label, score in best_label_for_box:
                box = box.int().tolist()
                label_name = rev_label_map[label.item()]
                cv2.rectangle(face_roi, (box[0], box[1]), (box[2], box[3]), label_color_map[label_name], 2)
                cv2.putText(face_roi, f"{label_name} ({score:.2f})", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            label_color_map[label_name], 2)
    # 计算并显示FPS
    current_time = time.time()
    delta_time = current_time - last_time
    last_time = current_time
    fps = 1.0 / delta_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # 显示结果
    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

