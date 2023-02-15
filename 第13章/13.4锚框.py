import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches    # 可以返回Rectangle对象，然后画到fig上
# from d2l import torch as d2l

torch.set_printoptions(2)


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


def multibox_prior(data, sizes, ratios):
    """ 生成以每个像素为中心的具有不同形状和大小的锚框 """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素中心，需要设置偏移量
    offset_h, offset_w = .5, .5
    step_h = 1. / in_height
    step_w = 1. / in_width

    center_h = (torch.arange(in_height, device=device) + offset_h) * step_h     # 果然，缩放到[0,1]之间
    center_w = (torch.arange(in_width, device=device) + offset_w) * step_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成所有锚框的高、宽，各boxes_per_pixel个，之后用于计算锚框的左上、右下坐标
    """
    代码里输入的r’是锚框的w/h，书里给的r是锚框的(w’/h’)除以原图的(w/h)，而代码里把r’当作r来算，最后为了满足w/h=r’，让w’乘以了 in_height / in_width.
实际上r = r’ * in_height / in_width. 和你下面的式子等价。当然两种方法都可以生成正确的锚框，就是逻辑和代码有出入。
    """
    w = torch.cat((size_tensor * torch.sqrt(ratios_tensor[0]), size_tensor[0] * torch.sqrt(ratios_tensor[1:]))) * in_height / in_width      # 此处乘以in_height可以理解，除以in_width似乎是为了归一化，但为什么h没有归一化？
    h = torch.cat((size_tensor / torch.sqrt(ratios_tensor[0]), sizes[0] / torch.sqrt(ratios_tensor[1:])))
    
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2    # 为了后面计算左上、右下坐标；每一行都是[-w,-h,w,h]
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)   # 每一行变成了[x,y,x,y]
    output = out_grid + anchor_manipulations    # [锚框数，4]
    return output.unsqueeze(0)      # 升维干什么


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """ 在axes上画出bboxes 
        后面要跟plt.show() """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, box in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], color=color, fill=False, linewidth=2)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """ 计算锚框列表或边界框列表中成对的交并比
        如：boxes1是[5,4]，boxes2是[5,4]，则计算过程是两两比较，结果5*5个box之间的iou """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])    # None升维——size[5,2]升为size[5,1,2]，为的是box之间两两能比较，计算结果为torch.Size([5, 5, 2]) 即5*5个box的计算结果
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # torch.Size([5, 5, 2]) 5*5个box的计算结果
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)    # 得到所有box的长、宽
    inter_areas = inters[:, :, 0] * inters[:, :, 1]     # 计算所有box的面积
    union_areas = areas1[:, None] + areas2 - inter_areas    # 同理，计算两两box之间的面积和，减去交集
    return inter_areas / union_areas


def assign_bbox_to_anchor(ground_truth, anchors, device, iou_threshold=.5):
    """ 将最接近的bbox分配给anchor """
    num_anchors, num_gt_bboxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)

    anchors_bbox_map = torch.full(size=(num_anchors, ), fill_value=-1, 
                                    dtype=torch.long, device=device)    # 初始化，尚未为anchor分配bbox
    max_ious, indices = torch.max(jaccard, dim=1)       # 按dim=1维度计算max值，即返回每一行的最大元素，同时返回索引（按行计算，当然返回列索引）
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)    # 哪些anchor能分配到bbox；返回(非零元素个数，参数维度[定位索引嘛])
    box_j = indices[max_ious >= iou_threshold].reshape(-1)      # 哪些bbox被分了出去
    anchors_bbox_map[anc_i] = box_j     # 行-第几个anchor  列-分配的第几个bbox，即第几个anchor分配到了第几个bbox；其余位置为-1
    col_discard = torch.full(size=(num_anchors, ), fill_value=-1)   # 用于后面分配好anchor-bbox矩阵后丢弃元素所在行列
    row_discard = torch.full(size=(num_gt_bboxes, ), fill_value=-1)
    for _ in range(num_gt_bboxes):      # 开始计算anchor-bbox矩阵，并丢弃已 分配了bbox的anchor 的行列
        max_idx = torch.argmax(jaccard)     # 从anchor-bbox矩阵选择有最大iou值的元素
        box_idx = (max_idx % num_gt_bboxes).long()      # max_ixd在anchor-bbox矩阵中排到第几列
        anc_idx = (max_idx / num_gt_bboxes).long()      # max_idx在anchor-bbox矩阵中排到第几行
        anchors_bbox_map[anc_idx] = box_idx     # 确定了目前最大iou元素的行列索引
        jaccard[:, box_idx] = col_discard       # 抛弃所在行、列
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """ 计算锚框偏移量 """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框
        参数进来之前都进行了升维——batch"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]      # anchors刚才squeeze(0)了，所以此时shape[0]便是数量
    for i in range(batch_size):
        label = labels[i, :, :]     # 维度i：batch
        anchors_bbox_map = assign_bbox_to_anchor(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1    # 分配给锚框的类别标签，记得+1, 0是背景类
        assigned_bb[indices_true] = label[bb_idx, 1:]       # 分配给锚框的bbox坐标
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """ 使用锚框和预测的偏移量，生成模型预测的边界框 """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """ nms非极大值抑制
        返回保留的预测框索引 """
    B = torch.argsort(scores, dim=-1, descending=True)      # 对所有预测框置信度降序排序，并返回索引
    keep = []
    while B.numel() > 0:    # 循环进行nms。这里的思路是一种，还有另一种思路的nms
        i = B[0]    # 取最大置信度的索引
        keep.append(i)      # 当前最大置信度索引，这个框得保留
        if B.numel() == 1:
            break
        iou = box_iou(boxes1=boxes[i, :].reshape(-1, 4), 
                        boxes2=boxes[B[1:], :].reshape(-1, 4)).reshape(-1)  # 计算B[0]代表的预测框和其他所有预测框的iou
        inds = torch.nonzero(iou <= iou_threshold)      # iou超过阈值的，说明太相似，丢弃，只保留不太相似的
        B = B[inds + 1]     # 这个+1是因为 刚才是B[0]和B[1:]计算的iou，返回的inds形状是[非零元素个数，参数形状--这里是1吧，毕竟iou是个数，同时也是索引，但是比B的索引小1，寻思寻思]
    return torch.tensor(keep, device=boxes.device)


#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

if __name__ == '__main__':
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                        [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                        [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    img = mimage.imread(f'images/001.png')
    w, h = img.shape[0], img.shape[1]
    bbox_scale = torch.tensor(data=(w, h, w, h))
    fig = plt.imshow(img)
    show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    plt.show()

    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                                offset_preds.unsqueeze(dim=0),
                                anchors.unsqueeze(dim=0),
                                nms_threshold=.5)

    # 一个例子
    # img = mimage.imread(f'images/001.png')
    # w, h = img.shape[0], img.shape[1]
    # bbox_scale = torch.tensor(data=(w, h, w, h))
    # ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
    #                      [1, 0.55, 0.2, 0.9, 0.88]])
    # anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
    #                     [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
    #                     [0.57, 0.3, 0.92, 0.9]])

    # fig = plt.imshow(img)
    # show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    # show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    # plt.show()

    # labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

    # img = mimage.imread(f'images/001.png')      # img.shape (140, 181, 3)
    # # plt.imshow(img)
    # # plt.show()
    # # print(multibox_prior(torch.rand(size=(1, 3, img.shape[0], img.shape[1])),
    # #                     sizes=(.75, .5, .25),
    # #                     ratios=(1, 2, .5)).shape)   # torch.Size([1, 126700, 4])
    # boxes = multibox_prior(torch.rand(size=(1, 3, img.shape[0], img.shape[1])), 
    #                         sizes=(.75, .5, .25),
    #                         ratios=(1, 2, .5))
    # boxes = boxes.reshape(img.shape[0], img.shape[1], 5, 4)
    # print(boxes[50, 50, 0, :])

    # w, h = img.shape[0], img.shape[1]
    # bbox_scale = torch.tensor(data=(w, h, w, h))
    # fig = plt.imshow(img)
    # show_bboxes(fig.axes, boxes[50, 50, :, :] * bbox_scale, ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
    # plt.show()

    # box_iou(boxes[40, 40, :, :], boxes[50, 50, :, :])

    # assign_bbox_to_anchor(boxes[20, 20, :, :], boxes[50, 50, :, :], boxes.device)

    