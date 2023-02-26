import cv2
import torch
import matplotlib.pyplot as plt
# from d2l import torch as d2l

torch.set_printoptions(2)   # 精简输出精度
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


def multibox_prior(data, sizes, ratios):
    """ 输入图像、缩放比、宽高比，返回所有生成的锚框torch.Size([1, 527250, 4]) """
    in_height, in_width = data.shape[-2:]    # 读入图片的维度就是B C H W了
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
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')    # https://pytorch.org/docs/stable/generated/torch.meshgrid.html
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成boxes_per_pixel个半高半宽
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), size_tensor[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), size_tensor[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2    # 重复 像素总数 遍--w、h是boxes_per_pixel个

    # 每个中心点都有boxes_per_pixel个锚框
    # torch.repeat_interleave https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave
    # torch.stack https://blog.csdn.net/xinjieyuan/article/details/105205326
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


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
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1], fill=False)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


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

def box_corner_to_center(boxes):
    """ 从13.3复制来的 """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return torch.stack((cx, cy, w, h), axis=-1)


def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


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





if __name__ == "__main__":
    # OpenCV读取图片
    img = cv2.imread('images/009.jpg', 0)   # h, w, c
    img = cv2.resize(img, (800, 600))
    # cv2.imshow("down sample", img)
    # cv2.waitKey(0)
    h, w = img.shape[:2]
    print(img.shape)
    X = torch.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[.75, .5, .25], ratios=[1, 2, .5])
    print(Y.shape)
    boxes = Y.reshape(h, w, 5, 4)
    print(boxes[250, 250, :, :])
    # print('iou:\n', box_iou(boxes[250, 250, :, :], boxes[230, 230, :, :]))

    bbox_scale = torch.tensor((w, h, w, h))
    fig = plt.imshow(img)
    # show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, labels=['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
    #                                                                   's=0.75, r=2', 's=0.75, r=0.5'])
    # ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
    #                              [1, 0.55, 0.2, 0.9, 0.88]])
    # anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
    #                         [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
    #                         [0.57, 0.3, 0.92, 0.9]])
    # show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    # show_bboxes(fig.axes, anchors * bbox_scale, list('01234'))
    # plt.show()

    # labels = multibox_target(anchors.unsqueeze(0), ground_truth.unsqueeze(0))
    # print(labels)

    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                              [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                              [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    # show_bboxes(fig.axes, anchors * bbox_scale, labels=['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    # plt.show()
    out = multibox_detection(cls_probs.unsqueeze(0), offset_preds.unsqueeze(0), anchors.unsqueeze(0), nms_threshold=.5)
    print(out)
    for i in out[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
    plt.show()

    # matplotlib读取图片
    # img = d2l.plt.imread('images/dog_cat.png')      # h, w, c
    # h, w = img.shape[:2]
    # print(img.shape)
    # X = torch.rand(size=(1, 3, h, w))
    # Y = multibox_prior(X, sizes=[.75, .5, .25], ratios=[1, 2, .5])
    # print(Y.shape)

