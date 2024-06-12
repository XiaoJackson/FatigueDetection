from typing import List, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=5, stride=self.stride, padding=2),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],#Block的重复次数
                 stages_out_channels: List[int],#c1,c5的channel
                 num_classes: int = 1000,#分类的类别个数
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels
        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2
        self.stage2 = self._make_stage(input_channels, self._stage_out_channels[1], 2, stages_repeats[0], inverted_residual)
        input_channels = self._stage_out_channels[1]

        # 1x1卷积层提升到512通道
        self.stage22 = nn.Conv2d(input_channels, 512, kernel_size=1, stride=1, padding=0)
        # 1x1卷积层还原到stage3的输入通道数
        self.stage22_1 = nn.Conv2d(512, self._stage_out_channels[2], kernel_size=1, stride=1, padding=0)

        # stage3
        self.stage3 = self._make_stage(256, self._stage_out_channels[2], 2, stages_repeats[1], inverted_residual)

        input_channels = 256
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stage(self, input_channels, output_channels, stride, repeats, inverted_residual):
        seq = [inverted_residual(input_channels, output_channels, stride)]
        for i in range(repeats - 1):
            seq.append(inverted_residual(output_channels, output_channels, 1))
        return nn.Sequential(*seq)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage22(x)
        stage2_features = x
        x = self.stage22_1(x)
        x = self.stage3(x)
        x = self.conv5(x)
        conv5_features = x

        return stage2_features,conv5_features


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def shufflenet_v2_SSD_backbone(num_classes=4):
    model = ShuffleNetV2(stages_repeats=[4, 4, 4],
                         stages_out_channels=[64, 128, 256, 512, 1024],
                         num_classes=num_classes)

    return model

model = shufflenet_v2_SSD_backbone()

class ExtraFeatureLayers(nn.Module):
    def __init__(self):
        super(ExtraFeatureLayers, self).__init__()

        # Define the extra layers
        self.conv1_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)  # 19x19x1024 -> 19x19x256
        self.conv1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 19x19x256 -> 10x10x512
        self.conv2_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)  # 10x10x512 -> 10x10x128
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 10x10x128 -> 5x5x256
        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)  # 5x5x256 -> 5x5x128
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 5x5x128 -> 3x3x256
        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)  # 3x3x256 -> 3x3x128
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)  # 3x3x128 -> 1x1x256

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv5_features):
        out = F.relu(self.conv1_1(conv5_features))
        out = F.relu(self.conv1_2(out))
        conv1_2_feats = out
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        conv2_2_feats = out
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        conv3_2_feats = out
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        conv4_2_feats = out
        return conv1_2_feats, conv2_2_feats, conv3_2_feats, conv4_2_feats

class PredictionConvolutions(nn.Module):
    """
    使用较低级别和较高级别的特征图来预测类别分数和边界框的卷积层。

    边界框（位置）被预测为相对于8732个先验（默认）框的编码偏移量。
    有关编码定义，请参见utils.py中的'cxcy_to_gcxgcy'。

    类别分数表示每个对象类别在8732个定位框中的得分。
    """

    def __init__(self, n_classes):
        """
        :param n_classes: 不同类型对象的数量
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # 每个特征图位置我们考虑的先验框数
        n_boxes = {'stage2': 4,
                   'conv5': 6,
                   'conv1_2': 6,
                   'conv2_2': 6,
                   'conv3_2': 4,
                   'conv4_2': 4}
        # 4个先验框意味着我们使用4个不同的宽高比等等。

        # 本地化预测卷积（预测相对于先验框的偏移量）
        self.loc_stage2 = nn.Conv2d(512, n_boxes['stage2'] * 4, kernel_size=3, padding=1)
        self.loc_conv5 = nn.Conv2d(1024, n_boxes['conv5'] * 4, kernel_size=3, padding=1)
        self.loc_conv1_2 = nn.Conv2d(512, n_boxes['conv1_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv2_2 = nn.Conv2d(256, n_boxes['conv2_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv3_2 = nn.Conv2d(256, n_boxes['conv3_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv4_2 = nn.Conv2d(256, n_boxes['conv4_2'] * 4, kernel_size=3, padding=1)

        # 类别预测卷积（在定位框中预测类别）
        self.cl_stage2 = nn.Conv2d(512, n_boxes['stage2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv5 = nn.Conv2d(1024, n_boxes['conv5'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv1_2 = nn.Conv2d(512, n_boxes['conv1_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv2_2 = nn.Conv2d(256, n_boxes['conv2_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv3_2 = nn.Conv2d(256, n_boxes['conv3_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv4_2 = nn.Conv2d(256, n_boxes['conv4_2'] * n_classes, kernel_size=3, padding=1)

        # 初始化卷积参数
        self.init_conv2d()

    def init_conv2d(self):
        """
        初始化卷积参数。
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, stage2_feats, conv5_feats, conv1_2_feats, conv2_2_feats, conv3_2_feats, conv4_2_feats):
        """
        前向传播。

        :param stage2_feats: stage2特征图，尺寸为（N，512，38，38）
        :param conv5_feats: conv5特征图，尺寸为（N，1024，19，19）
        :param conv1_2_feats: conv1_2特征图，尺寸为（N，512，10，10）
        :param conv2_2_feats: conv2_2特征图，尺寸为（N，256，5，5）
        :param conv3_2_feats: conv3_2特征图，尺寸为（N，256，3，3）
        :param conv4_2_feats: conv4_2特征图，尺寸为（N，256，1，1）
        :return: 每个图像的8732个位置和类别分数（即相对于每个先验框）。
        """
        batch_size = stage2_feats.size(0)

        # 预测定位框的边界（作为相对于先验框的偏移量）
        l_stage2 = self.loc_stage2(stage2_feats)  # （N，16，38，38）
        l_stage2 = l_stage2.permute(0, 2, 3, 1).contiguous()  # （N，38，38，16），以匹配先验框顺序（之后的.view()）
        # （.contiguous()确保它存储在连续的内存块中，需要用于下面的.view()）
        l_stage2 = l_stage2.view(batch_size, -1, 4)  # （N，5776，4），在此特征图上总共有5776个框

        l_conv5 = self.loc_conv5(conv5_feats)  # （N，24，19，19）
        l_conv5 = l_conv5.permute(0, 2, 3, 1).contiguous()  # （N，19，19，24）
        l_conv5 = l_conv5.view(batch_size, -1, 4)  # （N，2166，4），在此特征图上总共有2116个框

        l_conv1_2 = self.loc_conv1_2(conv1_2_feats)  # （N，24，10，10）
        l_conv1_2 = l_conv1_2.permute(0, 2, 3, 1).contiguous()  # （N，10，10，24）
        l_conv1_2 = l_conv1_2.view(batch_size, -1, 4)  # （N，600，4）

        l_conv2_2 = self.loc_conv2_2(conv2_2_feats)  # （N，24，5，5）
        l_conv2_2 = l_conv2_2.permute(0, 2, 3, 1).contiguous()  # （N，5，5，24）
        l_conv2_2 = l_conv2_2.view(batch_size, -1, 4)  # （N，150，4）

        l_conv3_2 = self.loc_conv3_2(conv3_2_feats)  # （N，16，3，3）
        l_conv3_2 = l_conv3_2.permute(0, 2, 3, 1).contiguous()  # （N，3，3，16）
        l_conv3_2 = l_conv3_2.view(batch_size, -1, 4)  # （N，36，4）

        l_conv4_2 = self.loc_conv4_2(conv4_2_feats)  # （N，16，1，1）
        l_conv4_2 = l_conv4_2.permute(0, 2, 3, 1).contiguous()  # （N，1，1，16）
        l_conv4_2 = l_conv4_2.view(batch_size, -1, 4)  # （N，4，4）

        # 在定位框中预测类别
        c_stage2 = self.cl_stage2(stage2_feats)  # （N，4 * n_classes，38，38）
        c_stage2 = c_stage2.permute(0, 2, 3, 1).contiguous()  # （N，38，38，4 * n_classes），以匹配先验框顺序（之后的.view()）
        c_stage2 = c_stage2.view(batch_size, -1, self.n_classes)  # （N，5776，n_classes），在此特征图上总共有5776个框

        c_conv5 = self.cl_conv5(conv5_feats)  # （N，6 * n_classes，19，19）
        c_conv5 = c_conv5.permute(0, 2, 3, 1).contiguous()  # （N，19，19，6 * n_classes）
        c_conv5 = c_conv5.view(batch_size, -1, self.n_classes)  # （N，2166，n_classes），在此特征图上总共有2116个框

        c_conv1_2 = self.cl_conv1_2(conv1_2_feats)  # （N，6 * n_classes，10，10）
        c_conv1_2 = c_conv1_2.permute(0, 2, 3, 1).contiguous()  # （N，10，10，6 * n_classes）
        c_conv1_2 = c_conv1_2.view(batch_size, -1, self.n_classes)  # （N，600，n_classes）

        c_conv2_2 = self.cl_conv2_2(conv2_2_feats)  # （N，6 * n_classes，5，5）
        c_conv2_2 = c_conv2_2.permute(0, 2, 3, 1).contiguous()  # （N，5，5，6 * n_classes）
        c_conv2_2 = c_conv2_2.view(batch_size, -1, self.n_classes)  # （N，150，n_classes）

        c_conv3_2 = self.cl_conv3_2(conv3_2_feats)  # （N，4 * n_classes，3，3）
        c_conv3_2 = c_conv3_2.permute(0, 2, 3, 1).contiguous()  # （N，3，3，4 * n_classes）
        c_conv3_2 = c_conv3_2.view(batch_size, -1, self.n_classes)  # （N，36，n_classes）

        c_conv4_2 = self.cl_conv4_2(conv4_2_feats)  # （N，4 * n_classes，1，1）
        c_conv4_2 = c_conv4_2.permute(0, 2, 3, 1).contiguous()  # （N，1，1，4 * n_classes）
        c_conv4_2 = c_conv4_2.view(batch_size, -1, self.n_classes)  # （N，4，n_classes）

        # 总共8732个框
        # 以特定顺序连接（即必须与先验框的顺序匹配）
        locs = torch.cat([l_stage2, l_conv5, l_conv1_2, l_conv2_2, l_conv3_2, l_conv4_2], dim=1)  # （N，8732，4）
        classes_scores = torch.cat([c_stage2, c_conv5, c_conv1_2, c_conv2_2, c_conv3_2, c_conv4_2],
                                   dim=1)  # （N，8732，n_classes）


        return locs, classes_scores


class ShuffleNetV2SSD(nn.Module):

    def __init__(self, n_classes):
        super(ShuffleNetV2SSD, self).__init__()

        self.n_classes = n_classes

        self.base = model  # 基础模型
        self.aux_convs = ExtraFeatureLayers()  # 额外的特征提取层
        self.pred_convs = PredictionConvolutions(n_classes)  # 预测卷积层

        # 由于低层特征（conv4_3_feats）具有相当大的尺度，我们采取L2范数并重新缩放
        # 重新缩放因子最初设置为20，但在反向传播期间会针对每个通道进行学习
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # conv4_3_feats中有512个通道
        nn.init.constant_(self.rescale_factors, 20)

        # 先验框
        self.priors_cxcy = self.create_prior_boxes()  # 创建先验框

    def forward(self, image):
        """
        前向传播。

        :param image: 图像，尺寸为 (N, 3, 300, 300) 的张量
        :return: 每张图像中针对每个先验框的 8732 个位置和类别分数
        """
        # 运行 Shufflenetv2 网络卷积（生成低层特征图）
        stage2_feats, conv5_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # 对 stage2 进行 L2 范数归一化后重新缩放
        norm = stage2_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        stage2_feats = stage2_feats / norm  # (N, 512, 38, 38)
        stage2_feats = stage2_feats * self.rescale_factors  # (N, 512, 38, 38)
        # （PyTorch 在算术运算期间自动广播单一维度）

        # 运行辅助卷积（生成更高级别的特征图）
        conv1_2_feats, conv2_2_feats, conv3_2_feats, conv4_2_feats = \
            self.aux_convs(conv5_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # 运行预测卷积（针对先验框预测相对于其偏移和每个结果定位框中的类别）
        locs, classes_scores = self.pred_convs(stage2_feats, conv5_feats, conv1_2_feats, conv2_2_feats, conv3_2_feats,
                                               conv4_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        创建 SSD300 中的 8732 个先验框（默认），如论文所定义。

        :return: 中心尺寸坐标中的先验框，维度为 (8732, 4) 的张量
        """
        fmap_dims = {'stage2': 38,
                     'conv5': 19,
                     'conv1_2': 10,
                     'conv2_2': 5,
                     'conv3_2': 3,
                     'conv4_2': 1}

        obj_scales = {'stage2': 0.1,
                      'conv5': 0.2,
                      'conv1_2': 0.375,
                      'conv2_2': 0.55,
                      'conv3_2': 0.725,
                      'conv4_2': 0.9}

        aspect_ratios = {'stage2': [1., 2., 0.5],
                         'conv5': [1., 2., 3., 0.5, .333],
                         'conv1_2': [1., 2., 3., 0.5, .333],
                         'conv2_2': [1., 2., 3., 0.5, .333],
                         'conv3_2': [1., 2., 0.5],
                         'conv4_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # 对于纵横比为 1 的情况，使用额外的先验框，其尺度为当前特征图尺度和下一个特征图尺度的几何平均值
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # 对于最后一个特征图，没有“下一个”特征图
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4); 此行代码无效；请参阅教程中的备注部分

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        解读 SSD300 输出的 8732 个位置和类别分数，以检测物体。

        对于每个类别，对得分高于某个最小阈值的框执行非最大抑制（NMS）。

        :param predicted_locs: 预测的位置/框与 8732 个先验框相关，维度为 (N, 8732, 4) 的张量
        :param predicted_scores: 每个编码位置/框的类别分数，维度为 (N, 8732, n_classes) 的张量
        :param min_score: 某个类别的框被认为匹配的最小阈值
        :param max_overlap: 两个框之间的最大重叠，以便分数较低的框不通过 NMS 抑制
        :param top_k: 如果所有类别的检测结果很多，则只保留前 'k' 个
        :return: 检测结果（框、标签和分数），长度为 batch_size 的列表
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # 存储所有图像的最终预测框、标签和分数的列表
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # 从回归到预测框形式解码对象坐标
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4)，这些是分数点坐标

            # 存储此图像的框和分数的列表
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # 针对每个类别进行检查
            for c in range(1, self.n_classes):
                # 仅保留得分高于最小分数的预测框和分数
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8（字节）张量，用于索引
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified)，n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # 按分数对预测框和分数进行排序
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified)，(n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # 找到预测框之间的重叠
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # 非最大抑制（NMS）

                # 一个 torch.uint8（字节）张量，用于跟踪要抑制的预测框
                # 1 表示抑制，0 表示不抑制
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # 按降序分数考虑每个框
                for box in range(class_decoded_locs.size(0)):
                    # 如果此框已标记为抑制
                    if suppress[box] == 1:
                        continue

                    # 抑制与此框的重叠大于最大重叠的框
                    # 找到这样的框并更新抑制索引
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # max 操作保留以前抑制的框，类似于 'OR' 操作

                    # 即使它与自身的重叠为 1，也不要抑制此框
                    suppress[box] = 0

                # 仅为此类别存储未抑制的框
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # 如果未发现任何类别中的对象，则为 'background' 存储占位符
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # 连接为单个张量
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # 仅保留前 k 个对象
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # 添加到存储所有图像的预测框和分数的列表
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # 长度为 batch_size 的列表


class MultiBoxLoss(nn.Module):
    """
    MultiBox 损失，用于目标检测的损失函数。

    这是以下内容的组合：
    (1) 对于预测框的位置的定位损失，和
    (2) 对于预测类别分数的置信度损失。
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # 文章中的 *smooth* L1 损失；请参阅教程中的备注部分
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        前向传播。

        :param predicted_locs: 预测的位置/框与 8732 个先验框相关，维度为 (N, 8732, 4) 的张量
        :param predicted_scores: 每个编码位置/框的类别分数，维度为 (N, 8732, n_classes) 的张量
        :param boxes: 边界坐标中的真实对象边界框，长度为 N 的张量列表
        :param labels: 真实对象标签，长度为 N 的张量列表
        :return: multibox 损失，标量
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # 对于每张图片
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # 对于每个先验框，找到具有最大重叠的对象
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # 我们不希望对象未在我们的正（非背景）先验框中表示 -
            # 1. 一个对象可能不是所有先验框的最佳对象，因此不在 object_for_each_prior 中。
            # 2. 所有具有该对象的先验框可能基于阈值（0.5）被分配为背景。

            # 为了解决这个问题 -
            # 首先，找到每个对象的具有最大重叠的先验框。
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # 然后，将每个对象分配给相应的具有最大重叠的先验框。 （这修复了 1。）
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # 为确保这些先验框符合要求，人为地给它们赋予了大于 0.5 的重叠度。（这修复了 2。）
            overlap_for_each_prior[prior_for_each_object] = 1.

            # 每个先验框的标签
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # 将与对象重叠的先验框的置信度小于阈值的先验框设置为背景（无对象）
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # 存储
            true_classes[i] = label_for_each_prior

            # 将中心尺寸对象坐标编码为我们回归到的预测框的形式
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # 确定是正（对象/非背景）先验框
        positive_priors = true_classes != 0  # (N, 8732)

        # 定位损失

        # 定位损失仅针对正（非背景）先验框计算
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # 注意：使用 torch.uint8（字节）张量进行索引会在索引跨多个维度（N 和 8732）时使张量扁平化
        # 因此，如果 predicted_locs 的形状为 (N, 8732, 4)，predicted_locs[positive_priors] 将具有（总正数，4）

        # 置信度损失
        # 置信度损失在正先验框和每张图片中由难度最大（最难）的负先验框计算
        # 也就是，对于每张图片，
        # 我们将取最难的（neg_pos_ratio * n_positives）负先验框，即损失最大的地方
        # 这称为难负样本挖掘 - 它集中在每张图片中最难的负样本，并且还最小化了正负不平衡

        # 每张图片的正和难负先验框数量
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # 首先，找到所有先验框的损失
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # 我们已经知道哪些是正的先验框
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # 接下来，找到哪些是难负样本
        # 为此，仅对每张图片中的负先验框按损失降序排序，并取前 n_hard_negatives 个
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732)，忽略了正先验框（永远不会是最难的前 n_hard_negatives 个）
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732)，按降序排序
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # 如论文中所述，仅在正先验框上平均，尽管在正负先验框上计算
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # 总损失

        return conf_loss + self.alpha * loc_loss