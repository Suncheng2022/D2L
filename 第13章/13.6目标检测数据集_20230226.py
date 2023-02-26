"""
加油，即将遇到13.7节——跑一个完整的目标检测模型
"""
import os
import pandas as pd
import torch
import torchvision
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

data_dir = r"../data/banana-detection"
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
        # images.append(torchvision.io.read_image(
        #     os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', img_name)
        # ))      # RGB
        img = cv2.imread(os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', img_name))    # hwc BGR，pytorch上计算需要改为cwh、RGB
        images.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
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


import matplotlib.patches as patches  # 可以返回Rectangle对象，然后画到fig上
# from d2l.torch import show_bboxes
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
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], color=color,
                                 fill=False, linewidth=2)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


if __name__ == '__main__':
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)       # torch.Size([32, 3, 256, 256]) torch.Size([32, 1, 5])

    # 展示10幅图片
    imgs = (batch[0][:10]) / 255        # 这里的 3 2，之前写颠倒了导致图片类似显示旋转了90°；read_data_bananas中没有对cv2读取的图片做permute，此处的permute倒也可以省去了
    fig, axes = plt.subplots(nrows=2, ncols=5)
    axes = axes.flatten()       # 将ax由n*m的Axes组展平成1*nm的Axes组
    for img, ax, label in zip(imgs, axes, batch[1][:10]):
        ax.imshow(img)
        show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()
