import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from utils import transform

new_labels = ('left_eye', 'right_eye', 'mouth')
label_map = {k: v + 1 for v, k in enumerate(new_labels)}  # 从1开始为每个新标签赋值
label_map['background'] = 0  # 背景标签，映射为0


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images_data = []
    objects_data = []

    for image in root.findall('image'):
        image_data = image.get('file')  # 使用图像的绝对路径
        objects = []
        for box in image.findall('box'):
            obj_dict = {
                'label': box.get('label'),
                'bbox': [
                    int(box.get('xmin')),
                    int(box.get('ymin')),
                    int(box.get('xmax')),
                    int(box.get('ymax'))
                ]
            }
            objects.append(obj_dict)
        images_data.append(image_data)
        objects_data.append(objects)

    return images_data, objects_data



class Dataset():
    def __init__(self, xml_path):
        self.images = []
        self.objects = []

        # 直接解析XML文件
        images_data, objects_data = parse_xml(xml_path)
        self.images.extend(images_data)  # 存储所有图片路径
        self.objects.extend(objects_data)  # 存储所有对象信息

        assert len(self.images) == len(self.objects), "Image and object counts should be the same"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 加载图像
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')

        # 获取这个图像的所有对象
        objects = self.objects[index]
        boxes = torch.tensor([obj['bbox'] for obj in objects], dtype=torch.float)  # 边界框列表
        labels = [label_map[obj['label']] for obj in objects]
        labels = torch.tensor(labels, dtype=torch.long)  # 标签列表

        # 应用图像转换
        image, boxes, labels = transform(image, boxes, labels)

        return image, boxes, labels

    def collate_fn(self, batch):
        """
        由于每张图片可能包含不同数量的物体，需要一个合并函数（collate function）传递给 `DataLoader` 使用。
        这个函数描述了如何组合这些不同大小的张量。我们使用列表来完成这个任务。
        注意：这个函数不一定要在当前类中定义，它可以是一个独立的函数。
        :param batch: 一个由 `__getitem__()` 返回的 N 组数据的可迭代对象
        :return: 一个图片的张量，包含不同大小的边界框张量列表、标签和难度的列表

        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels # tensor (N, 3, 300, 300), 3 lists of N tensors each
