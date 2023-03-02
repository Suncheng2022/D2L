import torch
import torchvision
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """ 渲染bboxes """
    def _make_list(obj, default_value=None):
        if not obj:
            obj = default_value
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, list('bgrmc'))
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1], fill=False, edgecolor=color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def offset_inverse(anchors, offset_preds):
    """计算 锚框 + 预测偏移量 """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshod):
    """ 对 预测边界框=锚框+偏移量 的置信度进行非极大值抑制，返回保留的预测框索引 """
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel():
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshod).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=.5, pos_threshold=.00999999):
    """

    :param cls_probs: [batch, num_classes, num_anchors]
    :param offset_preds: [batch,
    :param anchors: [batch,
    :param nms_threshold:
    :param pos_threshold:
    :return:
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)    # 偏移量有4个值
        conf, class_id = torch.max(cls_prob[1:], dim=0)     # [1:] 索引0是背景，不参加计算，dim=0是按行计算
        predicted_bb = offset_inverse(anchors, offset_pred)     # anchors是所有的，offset_pred是batch中的1个样本，可能写错了？
        keep = nms(predicted_bb, conf, nms_threshold)       # 对预测结果进行nms，返回保留的预测框索引

        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))       # 保留的预测框索引 和 所有的预测框索引 拼接在一起，后面通过计算索引出现次数就知道是否保留了
        uniques, counts = combined.unique(return_counts=True)      # 计算出现次数
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))     # 保留的预测框 和 不保留的预测框 拼在一起
        class_id[non_keep] = -1     # 不保留的预测框 的 预测类别 置为-1
        class_id = class_id[all_id_sorted]      # class_id 顺序为保留的、不保留的，并且不保留的预测类别为-1
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]   # conf, predicted_bb顺序也按照 保留的、不保留的排好
        below_min_idx = (conf < pos_threshold)      # 对其中保留下来的再做一次后处理，预测阈值没超过pos_threshold的丢弃--预测类别也置为-1
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]       # 本来conf[below_min_idx]预测置信度都较小，用1减似乎调大了？

        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


def multibox_prior(data, sizes, ratios):
    """
    from 13.4
    输入图像、缩放比、宽高比，返回所有生成的锚框torch.Size([1, 527250, 4])
    """
    in_height, in_width = data.shape[-2:]  # 读入图片的维度就是B C H W了
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 因为要将锚框中心放到像素上，设置锚框中心偏移量
    offset_h, offset_w = .5, .5
    steps_h = 1. / in_height
    steps_w = 1. / in_width

    # 生成所有锚框中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w,
                                      indexing='ij')  # https://pytorch.org/docs/stable/generated/torch.meshgrid.html
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成boxes_per_pixel个半高半宽
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), size_tensor[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width,
                                                                1) / 2  # 重复 像素总数 遍--w、h是boxes_per_pixel个

    # 每个中心点都有boxes_per_pixel个锚框
    # torch.repeat_interleave https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave
    # torch.stack https://blog.csdn.net/xinjieyuan/article/details/105205326
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

data_dir = r'../data/banana-detection'
def read_data_bananas(is_train=True):
    """
    读取香蕉检测数据集，并返回图片列表images、标签列表targets二者的tensor
    :param is_train:
    :return:
    """
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in tqdm(csv_data.iterrows()):
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', img_name)
        ))      # output (Tensor[image_channels, image_height, image_width]) 且RGB
        # img = cv2.imread(os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', img_name))    # hwc BGR，pytorch上计算需要改为cwh、RGB
        # images.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        # images.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))

        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananaDataset(torch.utils.data.Dataset):
    """ 自定义数据集类 """
    def __init__(self, is_train=True):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {len(self.features)} {"training examples" if is_train else "validation examples"}')

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False), batch_size)
    return train_iter, val_iter


def box_corner_to_center(boxes):
    """ 从13.3复制来的 """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return torch.stack((cx, cy, w, h), axis=-1)


def box_iou(boxes1, boxes2):
    """ 计算两个boxes列表的iou, 返回形状[boxes1数量,boxes2数量] """
    # boxes1 2的形状 (boxes数量,4)
    box_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 计算所有box面积
    areas1 = box_area(boxes1)       # 形状(boxes数量,)
    areas2 = box_area(boxes2)
    # 计算 boxes1的每个框都和boxes2的每个框 左上、右下坐标
    inter_upleft = torch.max(boxes1[:, None, :2], boxes2[:, :2])    # [boxes数量,2]-->[boxes数量,1,2] 两两计算：boxes1的每个框都和boxes2所有框计算
    inter_lowright = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 形状[boxes1数量,boxes2数量,2]
    # 计算boxes1和boxes2相交、并面积
    inter = torch.clamp(inter_lowright - inter_upleft, min=0)   # 得到所有box的宽、高，形状[boxes1数量,boxes2数量,2]
    inter_areas = inter[:, :, 0] * inter[:, :, 1]   # 形状[boxes1数量,boxes2数量]
    union_areas = areas1[:, None] + areas2 - inter_areas    # 升维，是为了areas1的每个元素都与areas2发生计算
    return inter_areas / union_areas    # 形状[boxes1数量,boxes2数量]


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=.5):
    """ 将最接近的gt分配给anchor  返回anchors_bbox_map 形状[num_anchors] """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)    # 形状[num_anchors, num_gt_boxes]
    anchors_bbox_map = torch.full(size=(num_anchors, ), fill_value=-1, device=device, dtype=torch.long)     # 形状[num_anchors, ] 索引代表第几个anchor，值代表分配到了第几个bbox
    max_ious, indices = torch.max(jaccard, dim=1)   # 二者形状[jaccard.shape[0]即anchors数]  indices是分配到的gt的索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)    # nonzero()找到iou大于阈值的anchors索引，既然是索引就可以reshape方便后面直接索引anchor
    box_j = indices[max_ious >= iou_threshold]      # indices是分配给anchor的gt索引——分配的条件是iou大于阈值
    anchors_bbox_map[anc_i] = box_j

    col_discard = torch.full(size=(num_anchors, ), fill_value=-1)
    row_discard = torch.full(size=(num_gt_boxes, ), fill_value=-1)
    for _ in range(num_gt_boxes):   # 所有的gt是否都分配了出去——因为上面只是分配了anchor与bbox的iou大于阈值的，gt有可能有剩余
        max_idx = torch.argmax(jaccard)     # 将jaccard抻平，然后返回最大值的索引
        box_idx = (max_idx % num_gt_boxes).long()   # max_idx落在哪一列
        anc_idx = (max_idx / num_gt_boxes).long()   # max_idx落在哪一行
        anchors_bbox_map[anc_idx] = box_idx     # 可能之前填过，不过这里填也不会重复，我想是的
        jaccard[:, box_idx] = col_discard   # 抛弃当前最大iou所在行列
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map     # 形状[num_anchors, ]


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """ 对分配到gt的anchor的坐标偏移量进行转换，有利于拟合. 返回xywh """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset



def multibox_target(anchors, labels):
    """ 使用gt标记anchors """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)     # 形状[anchors数量] 其索引表示第几个anchor
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)    # 重复4为了后面操作坐标置零
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)    # anchor分配的gt的类别
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)     # anchor分配的gt坐标

        indices_true = torch.nonzero(anchors_bbox_map >= 0)     # 形状[几个非零元素，anchors数量]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        offset = offset_boxes(anchors, assigned_bb) * bbox_mask     # bbox_mask将没分配到gt的anchor坐标置零，因为他们是负类
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


"""*************************************** 以下为SSD相关实现 ******************************************"""

def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    类别预测层:
        1.不使用全连接层输出，降低模型复杂度
        2.使用3×3conv，padding=1 不改变输出高宽，但输出通道表示类别--num_anchors * (num_classes + 1)
    比如输出是torch.Size([2, 33, 10, 10])，33是由num_anchors * (num_classes + 1)计算得来，则33*10*10即每个像素有33个分类结果
    :param num_inputs: 输入通道数
    :param num_anchors:
    :param num_classes:
    :return:
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    """
    预测边界框：
        为每个锚框预测4个偏移量，所以输出通道是num_inputs * 4
    :param num_inputs: 输入通道数
    :param num_anchors:
    :return:
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


def flatten_pred(pred):
    """
    将分类预测层结果展平，与不同尺度的预测结果拼接以高效计算\n
    :param pred: 形状如 torch.Size([2, 33, 10, 10])
    :return:
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)  # start_dim 从哪个维度开始flatten成一条线


def concat_preds(preds):
    """
    对多个层的分类预测结果 1.展平 2.拼接 \n
    展平的目的是，尽管不同层的类别预测结果在 C H W维度不等，我们仍然可以连接这两个预测输出.\n
    :param preds: 多个分类预测层的列表
    :return:
    """
    return torch.cat([flatten_pred(pred) for pred in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    """
    高、宽减半块。由2个3×3conv padding=1 + maxpool(2)组成，扩大感受野
    :param in_channels:
    :param out_channels:
    :return: 注意返回需要用nn.Sequential()包装
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*blk)


def base_net():
    """
    基本网络块：即3个down_sample_blk
        1.经过base_net()，通道增加，高宽减小
        2.输入一定为3通道，已写死；base_net()是网络的浅层，直接处理图像
    :return:
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """
    可分别获取完整的模型的各块
    :param i:
    :return: 返回模型的某块
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)  # base_net最后一层输出64通道；宽高减半，通道翻倍
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    else:
        blk = down_sample_blk(128, 128)  # 期间的块，高宽减半、输入输出通道不变
    return blk


def blk_forward(X, blk, sizes, ratios, cls_predictor, bbox_predictor):
    """ 定义 模块/层 的计算 """
    Y = blk(X)
    anchors = multibox_prior(Y, sizes, ratios)  # 为特征图生成各种不同大小、形状的锚框
    cls_preds = cls_predictor(Y)  # 参数没问题？
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    """
    SSD完整模型:\n
        1.定义各模块/层 及 计算结果变量
        2.计算各模块/层，返回结果
    """

    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]  # TinySSD有5个块，各块的输出通道数
        # 定义TinySSD每一层及预测生成的类别和偏移量
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # 进行某模块的计算
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                                                                     getattr(self, f'cls_{i}'),
                                                                     getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)     # [batch, ]
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)     # [batch, , num_classes]
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds   # torch.Size([1, 5444, 4]) torch.Size([32, 5444, 2]) torch.Size([32, 21776])


if __name__ == '__main__':
    # 测试读一张图片
    # imgPath = "images/002.png"
    # imgCV2 = cv2.imread(imgPath)    # h w c
    # print(imgCV2.shape)
    #
    #
    # imgTorch = torchvision.io.read_image(imgPath)   # c h w
    # print(imgTorch.shape)
    #
    # exit(-1)


    Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
    print(Y1.shape, Y2.shape)
    Y1_Y2 = concat_preds([Y1, Y2])
    print(Y1_Y2.shape)

    X = forward(torch.zeros(size=(2, 3, 20, 20)), down_sample_blk(3, 10))
    print(X.shape)

    X = forward(torch.zeros(size=(2, 3, 256, 256)), base_net())
    print(X.shape)

    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1

    # net = TinySSD(num_classes=1)
    # X = torch.zeros(size=(32, 3, 256, 256))     # 定义1个batch数据
    # anchors, cls_preds, bbox_preds = net(X)
    # print(f'output anchors: {anchors.shape}\n'
    #       f'output cls_preds: {cls_preds.shape}\n'
    #       f'output bbox_preds: {bbox_preds.shape}')


    # 预测目标
    device = torch.device(f'cuda:0') if torch.cuda.device_count() else torch.device(f'cpu')

    # X = torchvision.io.read_image('../data/banana-detection/bananas_val/images/0.png').unsqueeze(0).float()
    X = torch.from_numpy(cv2.imread('../data/banana-detection/bananas_val/images/0.png')).permute(2, 0, 1).unsqueeze(0).float()
    # img = X.squeeze(0).permute(1, 2, 0).long()      # 这不刚好是cv2读的维度
    img = cv2.imread('../data/banana-detection/bananas_val/images/0.png')       # 这不刚好是cv2读的维度
    net = TinySSD(num_classes=1)
    net = net.to(torch.device(f'cuda'))
    def predict(X):
        net.eval()
        net.load_state_dict(torch.load('runs/SSD/epoch_5.pth'))
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]

    def display(img, output, threshold):
        fig = plt.imshow(img)
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            h, w = img.shape[:2]
            bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
            show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

    output = predict(X) # gpu
    display(img, output.cpu(), threshold=.1)

    exit(-1)

    # 训练模型
    from d2l import torch as d2l
    batch_size = 16
    train_iter, _ = load_data_bananas(batch_size)

    device = torch.device(f'cuda:0') if torch.cuda.device_count() else torch.device(f'cpu')
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=.2, weight_decay=5e-4)

    # https://zh.d2l.ai/chapter_computer-vision/ssd.html#id14
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        cls = cls_loss(cls_preds.reshape(-1, num_classes),
                       cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bbox_preds * bbox_masks,
                         bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox

    def cls_eval(cls_preds, cls_labels):
        """ 类别准确率评价 cls_eval torch.Size([16, 5444, 2])  cls_labels torch.Size([16, 5444])"""
        return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum()) / cls_labels.numel()
    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        """ 预测偏移量评价 """
        return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()) / bbox_labels.numel()

    num_epochs = 20
    net = net.to(device)
    for epoch in range(num_epochs):
        net.train()
        for i, (features, target) in enumerate(train_iter):
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            print(f'epoch {epoch + 1}/{num_epochs}: iter {i + 1}/{len(train_iter)} total loss:{l.sum():.4f}\t'
                  f'cls acc:{cls_eval(cls_preds, cls_labels) * 100:.4f}% \tbbox_pred acc:{bbox_eval(bbox_preds, bbox_labels, bbox_masks) * 100:.4f}%')
        torch.save(net.state_dict(), f'runs/SSD/epoch_{epoch + 1}.pth')

