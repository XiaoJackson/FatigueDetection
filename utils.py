import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义新的标签
new_labels = ('left_eye', 'right_eye', 'mouth')
label_map = {k: v + 1 for v, k in enumerate(new_labels)}  # 从1开始为每个新标签赋值

label_map['background'] = 0  # 背景标签，映射为0


# 反向映射，从整数值映射回标签名
rev_label_map = {v: k for k, v in label_map.items()}
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # 移除开头的 `#` 字符
    # 将十六进制颜色分割成 RGB 三个部分，并转换为整数
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

# 更新 distinct_colors 为 RGB 列表
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#FFFFFF']
rgb_colors = [hex_to_rgb(color) for color in distinct_colors]

# 使用更新后的 RGB 颜色列表更新 label_color_map
label_color_map = {k: rgb_colors[i] for i, k in enumerate(label_map.keys())}




def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    创建图像列表、图像中对象的边界框和标签，并将其保存到文件中。

    :param voc07_path: 'VOC2007'文件夹的路径
    :param voc12_path: 'VOC2012'文件夹的路径
    :param output_folder: 必须保存JSON文件的文件夹
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # 训练数据
    for path in [voc07_path, voc12_path]:

        # 查找训练数据中的图像ID
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # 解析注释的XML文件
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # 保存到文件
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # 也保存标签映射

    print('\n有 %d 张训练图像，共含有 %d 个对象。文件已保存到 %s。' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # 测试数据
    test_images = list()
    test_objects = list()
    n_objects = 0

    # 查找测试数据中的图像ID
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # 解析注释的XML文件
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # 保存到文件
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\n有 %d 张测试图像，共含有 %d 个对象。文件已保存到 %s。' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))



def decimate(tensor, m):
    """
    将张量按因子 'm' 减少，即通过保留每个 'm' 个值来进行下采样。

    当我们将全连接层转换为等效的尺寸较小的卷积层时使用此方法。

    :param tensor: 要减少的张量
    :param m: 张量各维度的减少因子列表；如果不沿某个维度减少，则为 None
    :return: 减少后的张量
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    计算检测到的对象的平均精度（mAP）。

    请参阅 https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 进行解释

    :param det_boxes: 包含检测到的对象边界框的每个图像的张量列表，每个张量表示一个图像
    :param det_labels: 包含检测到的对象标签的每个图像的张量列表，每个张量表示一个图像
    :param det_scores: 包含检测到的对象标签分数的每个图像的张量列表，每个张量表示一个图像
    :param true_boxes: 包含实际对象边界框的每个图像的张量列表，每个张量表示一个图像
    :param true_labels: 包含实际对象标签的每个图像的张量列表，每个张量表示一个图像
    :param true_difficulties: 包含实际对象难度（0 或 1）的每个图像的张量列表，每个张量表示一个图像
    :return: 所有类别的平均精度列表，平均精度（mAP）
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # 这些都是相同长度的张量列表，即图像数

    n_classes = len(label_map)

    # 将所有（真实）对象存储在一个连续的张量中，并跟踪它来自的图像
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects 是所有图像中的对象总数
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # 将所有检测结果存储在一个连续的张量中，并跟踪它来自的图像
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # 为每个类别（不包括背景）计算 AP
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # 仅提取具有该类别的对象
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # 忽略难对象

        # 跟踪已经被“检测到”的具有该类别的真实对象
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # 仅提取具有该类别的检测结果
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # 根据置信度/分数的降序对检测结果进行排序
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # 按照分数降序的顺序，检查真阳性或假阳性
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # 查找同一图像中具有该类别的对象、它们的难度以及之前是否已经检测到
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # 如果该图像中没有这样的对象，则检测结果是假阳性
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # 查找此检测结果与该类别图像中对象的最大重叠
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' 是这些图像级张量 'object_boxes'、'object_difficulties' 中的对象索引
            # 在原始类别级张量 'true_class_boxes' 等中，'ind' 对应于具有索引的对象...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # 我们需要 'original_ind' 来更新 'true_class_boxes_detected'

            # 如果最大重叠大于阈值 0.5，则匹配
            if max_overlap.item() > 0.5:
                # 如果匹配对象是“困难”的，则忽略
                if object_difficulties[ind] == 0:
                    # 如果尚未检测到此对象，则为真阳性
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # 现在已检测到/已考虑此对象
                    # 否则，它是假阳性（因为此对象已经被考虑）
                    else:
                        false_positives[d] = 1
            # 否则，检测结果与实际对象的位置不同，是假阳性
            else:
                false_positives[d] = 1

        # 在分数降序的顺序中计算累积精度和召回率
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # 找到大于阈值 't' 的召回率对应的精度的最大平均值
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c 在 [1, n_classes - 1] 中

    # 计算平均精度（mAP）
    mean_average_precision = average_precisions.mean().item()

    # 在字典中保留每个类别的平均精度
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    将边界坐标（x_min、y_min、x_max、y_max）的边界框转换为中心-尺寸坐标（c_x、c_y、w、h）。

    :param xy: 边界坐标中的边界框，大小为 (n_boxes, 4) 的张量
    :return: 中心-尺寸坐标中的边界框，大小为 (n_boxes, 4) 的张量
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h



def cxcy_to_xy(cxcy):
    """
    将中心-尺寸坐标（c_x、c_y、w、h）的边界框转换为边界坐标（x_min、y_min、x_max、y_max）。

    :param cxcy: 中心-尺寸坐标中的边界框，大小为 (n_boxes, 4) 的张量
    :return: 边界坐标中的边界框，大小为 (n_boxes, 4) 的张量
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    将边界框（以中心-尺寸形式表示）相对于相应的先验框（以中心-尺寸形式表示）进行编码。

    对于中心坐标，找到相对于先验框的偏移量，并按先验框的尺寸进行缩放。
    对于尺寸坐标，按先验框的尺寸进行缩放，并转换为对数空间。

    在模型中，我们正在预测以这种编码形式的边界框坐标。

    :param cxcy: 中心-尺寸坐标中的边界框，大小为 (n_priors, 4) 的张量
    :param priors_cxcy: 用于进行编码的先验框，大小为 (n_priors, 4) 的张量
    :return: 编码后的边界框，大小为 (n_priors, 4) 的张量
    """

    # 下面的 10 和 5 被称为原始 Caffe 仓库中的 'variances'，完全是经验性的
    # 它们用于某种数值调整，用于 'scaling the localization gradient'
    # 参见 https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h



def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    解码模型预测的边界框坐标，因为它们是使用上述形式进行编码的。

    它们被解码为中心-尺寸坐标。

    这是上面函数的逆操作。

    :param gcxgcy: 编码的边界框，即模型的输出，大小为 (n_priors, 4) 的张量
    :param priors_cxcy: 定义编码的先验框，大小为 (n_priors, 4) 的张量
    :return: 解码后的中心-尺寸形式的边界框，大小为 (n_priors, 4) 的张量
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h



def find_intersection(set_1, set_2):
    """
    找到处于边界坐标的两组框之间的每个框组合的交集。

    :param set_1: 第一组，维度为 (n1, 4) 的张量
    :param set_2: 第二组，维度为 (n2, 4) 的张量
    :return: 第一组中每个框与第二组中每个框的交集，维度为 (n1, n2) 的张量
    """

    # PyTorch 会自动广播单例维度
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)



def find_jaccard_overlap(set_1, set_2):
    """
    找到处于边界坐标的两组框之间的每个框组合的 Jaccard 重叠（IoU）。

    :param set_1: 第一组，维度为 (n1, 4) 的张量
    :param set_2: 第二组，维度为 (n2, 4) 的张量
    :return: 第一组中每个框与第二组中每个框的 Jaccard 重叠，维度为 (n1, n2) 的张量
    """

    # 找到交集
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # 找到两组中每个框的面积
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # 找到并集
    # PyTorch 会自动广播单例维度
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    通过将图像放置在填充材料的较大画布中执行放大操作。

    有助于学习检测较小的对象。

    :param image: 图像，维度为 (3, original_h, original_w) 的张量
    :param boxes: 边界坐标的边界框，维度为 (n_objects, 4) 的张量
    :param filler: 填充材料的 RGB 值，类似于 [R, G, B] 的列表
    :return: 扩展后的图像，更新后的边界框坐标
    """
    # 计算拟议扩展（缩放）图像的尺寸
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # 使用填充创建这样的图像
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # 注意 - 不要使用 expand()，如 new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # 因为所有扩展的值将共享相同的内存，因此更改一个像素将更改所有像素

    # 将原始图像随机放置在此新图像中的随机坐标处（图像的左上角为原点）
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # 相应地调整边界框的坐标
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects 是图像中的对象数量

    return new_image, new_boxes



def random_crop(image, boxes, labels):
    """
    执行论文中所述的随机裁剪。有助于学习检测较大和部分对象。

    请注意，有些对象可能会被完全裁剪掉。

    改编自 https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: 图像，维度为 (3, original_h, original_w) 的张量
    :param boxes: 边界坐标的边界框，维度为 (n_objects, 4) 的张量
    :param labels: 对象的标签，维度为 (n_objects) 的张量
    :param difficulties: 对这些对象的检测困难程度，维度为 (n_objects) 的张量
    :return: 裁剪后的图像，更新后的边界框坐标，更新后的标签，更新后的困难程度
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # 保持选择最小重叠值，直到成功裁剪
    while True:
        # 随机选择最小重叠值
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' 指不裁剪

        # 如果不裁剪
        if min_overlap is None:
            return image, boxes, labels

        # 尝试最多 50 次以此最小重叠值
        # 这在论文中没有提到，但在作者的原始 Caffe 仓库中选择了 50
        max_trials = 50
        for _ in range(max_trials):
            # 裁剪尺寸必须在原始尺寸的 [0.3, 1] 范围内
            # 注意 - 论文中是 [0.1, 1]，但实际上是在作者的仓库中是 [0.3, 1]
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # 纵横比必须在 [0.5, 2] 范围内
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # 裁剪坐标（图像左上角为原点）
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # 计算裁剪与边界框之间的 Jaccard 重叠
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects)，n_objects 是图像中的对象数量
            overlap = overlap.squeeze(0)  # (n_objects)

            # 如果没有一个边界框的 Jaccard 重叠大于最小值，则重试
            if overlap.max().item() < min_overlap:
                continue

            # 裁剪图像
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # 找到原始边界框的中心
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # 找到边界框中心位于裁剪中的边界框
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects)，一个 Torch uInt8/Byte 张量，可以用作布尔索引

            # 如果没有一个边界框的中心在裁剪中，则重试
            if not centers_in_crop.any():
                continue

            # 丢弃不满足此条件的边界框
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # 计算裁剪中边界框的新坐标
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] 是 [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] 是 [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels



def flip(image, boxes):
    """
    水平翻转图像。

    :param image: 图像，一个 PIL 图像
    :param boxes: 边界坐标的边界框，维度为 (n_objects, 4) 的张量
    :return: 翻转后的图像，更新后的边界框坐标
    """
    # 翻转图像
    new_image = FT.hflip(image)

    # 翻转边界框
    new_boxes = boxes.clone()
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes



def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    调整图像大小。对于 SSD300，调整为 (300, 300)。

    由于在此过程中为边界框计算了百分比/分数坐标（相对于图像尺寸），您可以选择保留它们。

    :param image: 图像，一个 PIL 图像
    :param boxes: 边界坐标的边界框，维度为 (n_objects, 4) 的张量
    :param dims: 调整后的图像尺寸，默认为 (300, 300)
    :param return_percent_coords: 是否返回百分比坐标，默认为 True
    :return: 调整后的图像，更新后的边界框坐标（或分数坐标，在这种情况下保持不变）
    """
    # 调整图像大小
    new_image = FT.resize(image, dims)

    # 调整边界框
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # 百分比坐标

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes



def photometric_distort(image):
    """
    扭曲亮度、对比度、饱和度和色相，每种扭曲都有50%的概率，并以随机顺序。

    :param image: 图像，一个 PIL 图像
    :return: 扭曲后的图像
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image



def transform(image, boxes, labels):
    """
    应用上述的变换。

    :param image: 图像，一个 PIL 图像
    :param boxes: 边界坐标中的边界框，维度为 (n_objects, 4) 的张量
    :param labels: 对象的标签，维度为 (n_objects) 的张量
    :param difficulties: 这些对象的检测困难度，维度为 (n_objects) 的张量
    :param split: 'TRAIN' 或 'TEST' 中的一个，因为应用不同的变换集
    :return: 转换后的图像、转换后的边界框坐标、转换后的标签、转换后的困难度
    """

    # ImageNet 数据集的均值和标准差，我们的基础 VGG 模型来自 torchvision 的模型
    # 参考：https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    # 对于评估/测试，跳过以下操作
    # 按随机顺序应用一系列图像扭曲，每种扭曲的发生概率为50%，与 Caffe 仓库相同
    new_image = photometric_distort(new_image)

    # 将 PIL 图像转换为 Torch 张量
    new_image = FT.to_tensor(new_image)

    # 使用 50% 的概率扩展图像（放大）- 有助于训练检测小对象
    # 用基础 VGG 训练的 ImageNet 数据集的均值填充周围空间
    if random.random() < 0.5:
        new_image, new_boxes = expand(new_image, boxes, filler=mean)

    # 随机裁剪图像（缩小）
    new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

    # 将 Torch 张量转换为 PIL 图像
    new_image = FT.to_pil_image(new_image)

    # 以50%的概率翻转图像
    if random.random() < 0.5:
        new_image, new_boxes = flip(new_image, new_boxes)

    # 将图像调整大小为 (300, 300) - 这也将绝对边界坐标转换为它们的分数形式
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # 将 PIL 图像转换为 Torch 张量
    new_image = FT.to_tensor(new_image)

    # 通过 ImageNet 数据集的均值和标准差进行标准化，这是我们基础 VGG 模型的标准化方式
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels



def adjust_learning_rate(optimizer, scale):
    """
    通过指定因子缩放学习率。

    :param optimizer: 需要缩小学习率的优化器。
    :param scale: 用于乘以学习率的因子。
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("减小学习率。\n 新的学习率为 %f\n" % (optimizer.param_groups[1]['lr'],))



def accuracy(scores, targets, k):
    """
    计算top-k准确率，从预测和真实标签中得出。

    :param scores: 模型的分数
    :param targets: 真实标签
    :param k: top-k准确率中的k
    :return: top-k准确率
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D张量
    return correct_total.item() * (100.0 / batch_size)



def save_checkpoint(epoch, model, optimizer):
    """
    保存模型检查点。

    :param epoch: 当前轮次
    :param model: 模型
    :param optimizer: 优化器
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)



class AverageMeter(object):
    """
    跟踪度量指标的最新值、平均值、总和以及计数。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    裁剪在反向传播期间计算的梯度，以避免梯度爆炸。

    :param optimizer: 具有要裁剪梯度的优化器。
    :param grad_clip: 裁剪值。
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)