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


def eye_mouth_state_detect(face_roi, label_text, box, ort_session2, ort_session3):
    left_eye_state_label = ''
    right_eye_state_label = ''
    mouth_state_label = ''
    if 'left_eye' in label_text:
        # 提取、处理并推理眼睛状态
        left_eye_roi = face_roi[box[1].int():box[3].int(), box[0].int():box[2].int()]
        if left_eye_roi.size != 0:
            left_eye_resized = cv2.resize(left_eye_roi, (24, 24))
            left_eye_pil = Image.fromarray(cv2.cvtColor(left_eye_resized, cv2.COLOR_BGR2RGB))
            left_eye_pil = left_eye_pil.convert("L")
            left_eye_pil = np.array(left_eye_pil)  # 首先，将 PIL.Image 对象转换为 NumPy 数组
            left_eye_pil1 = left_eye_pil.astype(np.float32) / 255.0
            # left_eye_pil1 = np.transpose(left_eye_pil1, (2, 0, 1))  # 调整通道顺序：HWC -> CHW
            left_eye_pil1 = np.expand_dims(left_eye_pil1, axis=0)  # 添加批处理维度
            left_eye_pil1 = np.expand_dims(left_eye_pil1, axis=0)  # 添加批处理维度
            input_name2 = ort_session2.get_inputs()[0].name
            left_eye_outputs = ort_session2.run(None, {input_name2: left_eye_pil1})
            # left_eye_state = eye_model(left_eye_tensor)
            left_eye_state_label = 'open' if (left_eye_outputs[0][0, 0]) > 0.2 else 'closed'
            #cv2.putText(face_roi, left_eye_state_label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if 'right_eye' in label_text:
        # 提取、处理并推理眼睛状态
        right_eye_roi = face_roi[box[1].int():box[3].int(), box[0].int():box[2].int()]
        if right_eye_roi.size != 0:
            right_eye_resized = cv2.resize(right_eye_roi, (24, 24))
            right_eye_pil = Image.fromarray(cv2.cvtColor(right_eye_resized, cv2.COLOR_BGR2RGB))
            right_eye_pil = right_eye_pil.convert("L")
            right_eye_pil = np.array(right_eye_pil)  # 首先，将 PIL.Image 对象转换为 NumPy 数组
            right_eye_pil1 = right_eye_pil.astype(np.float32) / 255.0
            # right_eye_pil1 = np.transpose(right_eye_pil1, (2, 0, 1))  # 调整通道顺序：HWC -> CHW
            right_eye_pil1 = np.expand_dims(right_eye_pil1, axis=0)  # 添加批处理维度
            right_eye_pil1 = np.expand_dims(right_eye_pil1, axis=0)  # 添加批处理维度
            input_name2 = ort_session2.get_inputs()[0].name
            right_eye_outputs = ort_session2.run(None, {input_name2: right_eye_pil1})
            # left_eye_state = eye_model(left_eye_tensor)
            right_eye_state_label = 'open' if (right_eye_outputs[0][0, 0]) > 0.2 else 'closed'
            #cv2.putText(face_roi, right_eye_state_label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if 'mouth' in label_text:
        # 提取、处理并推理嘴巴状态
        mouth_roi = face_roi[box[1].int():box[3].int(), box[0].int():box[2].int()]
        if mouth_roi.size != 0:
            mouth_resized = cv2.resize(mouth_roi, (60, 60))
            mouth_pil = Image.fromarray(cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2RGB))
            mouth_pil = mouth_pil.convert("L")
            mouth_pil = np.array(mouth_pil)  # 首先，将 PIL.Image 对象转换为 NumPy 数组
            mouth_pil1 = mouth_pil.astype(np.float32) / 255.0
            mouth_pil1 = np.expand_dims(mouth_pil1, axis=0)  # 添加批处理维度
            mouth_pil1 = np.expand_dims(mouth_pil1, axis=0)  # 添加批处理维度
            input_name3 = ort_session3.get_inputs()[0].name
            mouth_outputs = ort_session3.run(None, {input_name3: mouth_pil1})
            # left_eye_state = eye_model(left_eye_tensor)
            mouth_state_label = 'yawn' if (mouth_outputs[0][0, 0]) > 0.5 else 'noyawn'
            #cv2.putText(face_roi, mouth_state_label, (int(box[0]), int(box[2] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return [left_eye_state_label, right_eye_state_label, mouth_state_label]



# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载面部检测模型
model = ShuffleNetV2SSD(n_classes=4)
model = model.to('cpu')
ort_session1 = onnxruntime.InferenceSession("ssd2feature.onnx")

# 加载眼睛状态模型
ort_session2 = onnxruntime.InferenceSession("eyemodel.onnx")
# 加载嘴巴模型
ort_session3 = onnxruntime.InferenceSession("mouthmodel.onnx")

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
# r'/home/pi/Desktop/fatigue detectiononnx/video/2-FemaleNoGlasses-Yawning.avi'
# r'F:\aria2\dataset\YawDD.rar\YawDD\YawDD dataset\Dash\Male\11-MaleGlasses.avi'
cap = cv2.VideoCapture(0)
last_yawn_start_time = None
threshold = 0.7
fps = 0.0
last_time = time.time()
# 初始化状态变量
eye_closed_count = 0
mouth_yawn_count = 0
eye_closed_threshold = 2  # 连续闭眼计数阈值
mouth_yawn_threshold = 3  # 连续闭嘴计数阈值
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
fatigue_state = 'Normal'
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
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        face_roi = frame[box[1]:box[3], box[0]:box[2]]
        face_resized = cv2.resize(face_roi, (300, 300))
        face_pil = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_pil = cv2.resize(face_pil, (300, 300))
        face_pil1 = np.transpose(face_pil, (2, 0, 1))  # 调整通道顺序：HWC -> CHW
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = mean[:, np.newaxis, np.newaxis]
        std = std[:, np.newaxis, np.newaxis]
        # 进行归一化操作
        face_pil1 = (face_pil1 / 255.0 - mean) / std
        face_pil1 = np.expand_dims(face_pil1, axis=0)  # 添加批处理维度、
        face_pil1 = face_pil1.astype(np.float32)
        input_name1 = ort_session1.get_inputs()[0].name
        face_outputs = ort_session1.run(None, {input_name1: face_pil1})
        face_outputs_tensors = [torch.tensor(o) for o in face_outputs]
        face_outputs_tensors = [torch.tensor(item).to('cuda') if not isinstance(item, torch.Tensor) else item.to('cuda')
                                for item in face_outputs_tensors]
        if len(face_outputs_tensors) == 2:
            predicted_locs, predicted_scores = face_outputs_tensors
        else:
            raise RuntimeError("Expected exactly two output tensors from the ONNX model.")

        # 从模型输出解析检测框和标签
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores,
                                                                 min_score=0.48,
                                                                 max_overlap=0.6, top_k=200)

        # 调整尺寸到原图
        original_dims = torch.FloatTensor(
            [face_roi.shape[1], face_roi.shape[0], face_roi.shape[1], face_roi.shape[0]]).unsqueeze(0).to(
            'cpu')
        det_boxes = det_boxes[0].to('cpu') * original_dims
        det_labels = det_labels[0].to('cpu')

        for box, label in zip(det_boxes, det_labels):
            label_text = rev_label_map[label.item()]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # 检查框坐标是否在图像尺寸范围内
            if x1 < 0 or y1 < 0 or x2 > face_roi.shape[1] or y2 > face_roi.shape[0] or x1 >= x2 or y1 >= y2:
                continue
            left_eye_state_label, right_eye_state_label, mouth_state_label = eye_mouth_state_detect(face_roi,
                                                                                                    label_text, box,
                                                                                                    ort_session2,
                                                                                                    ort_session3)
            #print(mouth_state_label)
            # 判断眼睛状态
            if left_eye_state_label == 'closed' or right_eye_state_label == 'closed':
                eye_closed_count += 1
            elif left_eye_state_label == '' and right_eye_state_label == '':
                # 当左右眼状态都为''时，计数器保持不变
                eye_closed_count = eye_closed_count
            else:
                eye_closed_count = max(0, eye_closed_count - 1)
            # 判断是否达到阈值
            if eye_closed_count >= eye_closed_threshold:
                eye_closed = True
            else:
                eye_closed = False
            # print(eye_closed_count)
            # 判断嘴巴状态
            # 判断嘴巴状态
            if mouth_state_label == 'yawn':
                mouth_yawn_count += 1
            else:
                mouth_yawn_count = max(0, mouth_yawn_count - 2)
            # 判断是否达到阈值
            if mouth_yawn_count >= mouth_yawn_threshold:
                mouth_yawn = True
            else:
                mouth_yawn = False
            #print(mouth_yawn_count)
            # print(eye_closed, mouth_yawning)
            if eye_closed or mouth_yawn:
                fatigue_state = 'Fatigued'
            else:
                fatigue_state = 'Normal'
            #print(fatigue_state)
            cv2.putText(frame, f"Fatigue State: {fatigue_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 225), 2)
    # 计算并显示FPS
    current_time = time.time()
    delta_time = current_time - last_time
    last_time = current_time
    print(f"time:{delta_time * 1000}")
    # fps = 1.0 / delta_time
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # 显示结果
    cv2.imshow('Fatigue Detection on Raspberry Pi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
