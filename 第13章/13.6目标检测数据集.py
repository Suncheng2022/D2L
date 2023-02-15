"""
尝试手动实现读取目标检测数据集代码
Tip:
    1.PIL读的图片需要用transforms转成tensor
"""
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils import data
from PIL import Image


def read_data_bananas(is_train=True):
    data_dir = r'D:\Projects\D2L\data\banana-detection'     # 强制不转义
    csv_name = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_name)
    csv_data = csv_data.set_index('img_name')   # 设置索引列 img_name label xmin ymin xmax ymax
    trans = transforms.ToTensor()
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        img = Image.open(os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', img_name))
        images.append(trans(img))      # 读为RGB三维张量
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataSet(data.Dataset):
    def __init__(self, is_train) -> None:
        self.features, self.labels = read_data_bananas(is_train)
        print(f'读取{len(self.features)}个{"training" if is_train else "validation"}样本')

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    
    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    train_iter = data.DataLoader(BananasDataSet(True), batch_size, shuffle=True)
    val_iter = data.DataLoader(BananasDataSet(False), batch_size)
    return train_iter, val_iter

if __name__ == '__main__':
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))      # 取到的是1个batch

    # 同样，我可以用PIL来把tensor保存下来
    trans = transforms.ToPILImage()
    for i in range(2 * 6):
        img = trans(batch[0][i])
        img.save(f'{i}.jpg')