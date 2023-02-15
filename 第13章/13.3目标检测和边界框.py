import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"   # 这两行影响显示图像

import torch
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt

# img = Image.open('images/dog_cat.png')

# plt.figure('demo')
# plt.imshow(img)
# plt.show()

def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return torch.stack((cx, cy, w, h), axis=-1)

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def bbox_to_rect(bbox, color):
    return plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=2)
    

if __name__ == '__main__':
    dog_bbox, cat_bbox = [35.0, 20.0, 240, 250], [200, 50, 330, 240]
    # print(box_corner_to_center(torch.tensor(data=(dog_bbox, cat_bbox))))
    # print(box_center_to_corner(box_corner_to_center(torch.tensor(data=(dog_bbox, cat_bbox)))))

    img = Image.open('images/dog_cat.png')
    plt.figure('demo')
    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    plt.show()