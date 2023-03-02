import torch
import matplotlib.pyplot as plt

torch.set_printoptions(precision=2)     # 计算结果保留两位小数，四舍五入

def bbox_corner_to_center(boxes):
    """
    :param boxes: size(n, 4) 可以是批量，也可以是单个bbox 但要符合批量维度的形式
    :return: size(n, 4)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    return torch.stack(tensors=(cx, cy, w, h), dim=-1)      # Concatenates a sequence of tensors along a new dimension.


def bbox_center_to_corner(boxes):
    """
    :param boxes:size(n, 4)  可以是批量，也可以是单个bbox 但要符合批量维度的形式
    :return: size(n, 4)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    return torch.stack(tensors=(x1, y1, x2, y2), dim=-1)    # 返回的变量顺序要格外注意，耗时！


def bbox_to_rect(bboxCorner, color):
    """
    :param bboxCorner: 左上右下坐标，但plt.Rectangle要左上、宽、高
    :param color:
    :return:
    """
    return plt.Rectangle(xy=(bboxCorner[0], bboxCorner[1]), width=bboxCorner[2] - bboxCorner[0],
                         height=bboxCorner[3] - bboxCorner[1], fill=False, edgecolor=color, linewidth=2)


def multibox_prior(data, sizes, ratios):
    """
    为输入图像data生成各种大小比例的anchors并返回。
    :param data: 输入图像[b, c, h, w]，只用到了h w属性，没有对图像作操作
    :param sizes:
    :param ratios:
    :return:
    """
    in_hight, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    sizes_tensor = torch.tensor(sizes, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = .5, .5
    step_h, step_w = 1 / in_hight, 1 / in_width

    center_h = (torch.arange(in_hight, device=device) + offset_h) * step_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * step_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')    # https://blog.csdn.net/weixin_39504171/article/details/106356977
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # h w for anchors, it's not identified with book here
    h = torch.cat((sizes_tensor / torch.sqrt(ratios_tensor[0]), sizes_tensor[0] / torch.sqrt(ratios_tensor[1:]))) * in_hight
    w = torch.cat((sizes_tensor * torch.sqrt(ratios_tensor[0]), sizes_tensor[0] * torch.sqrt(ratios_tensor[1:]))) * in_hight

    # w h的长度都是boxes_per_pixel，即每个像素都应该有boxes_per_pixel个(w, h, -w, -h)以生成boxes_per_pixel个不同的anchors
    anchors_manipulations = torch.stack((w, h, -w, -h)).T.repeat(in_hight * in_width, 1) / 2

    out_grid = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchors_manipulations
    return output.unsqueeze(0)


if __name__ == '__main__':
    # fig = plt.figure()    # plt.imshow()也可以返回fig，这句可以不写
    img = plt.imread(fname=r'images/catdog.jpg')    # h w c
    fig = plt.imshow(img)
    # plt.show()

    dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
    # boxes = torch.tensor(data=(dog_bbox, cat_bbox))
    # print(f'{boxes.shape}\n{bbox_center_to_corner(bbox_corner_to_center(boxes)) == boxes}')

    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'r'))
    # plt.show()

    multibox_prior(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0), [1, .5, .25], [1, .5, .25])
