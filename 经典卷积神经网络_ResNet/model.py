import torch
import torch.nn.functional as F

from torch import nn


class ResidualUnit(nn.Module):
    """
    建立一个残差单元：
        tips：需要接收strides参数，定义第一个残差单元的时候，其第一个卷积层步幅=1；后面的残差单元第一个卷积层步幅=2
            恒等映射线路的步幅要同步接收的strides参数，以匹配输出大小
    """

    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.b1 = nn.BatchNorm2d(out_channels)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.b1(self.conv1(X)))
        Y = self.b2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


# test
# # blk = Residual(in_channels=3, out_channels=3)
# # 使用了strides=2要同时use_1x1conv=True，残差线路大小减半了，恒等映射也得如此
# blk = Residual(in_channels=3, out_channels=6, strides=2, use_1x1conv=True)   # 高宽减半，通道翻倍
# X = torch.rand(size=(4, 3, 224, 224))
# X = blk(X)
# print(X.shape)

def resnet_block(in_channels, out_channels, num_blocks, first_block=False):
    """ 构造残差块--通常由2个残差单元组成 """
    blk = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            blk.append(ResidualUnit(in_channels, out_channels, strides=2, use_1x1conv=True))
        else:
            blk.append(ResidualUnit(out_channels, out_channels))
    return nn.Sequential(*blk)


class ResModel(nn.Module):
    def __init__(self, block_arch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
            ),
            resnet_block(*block_arch[0]),
            resnet_block(*block_arch[1]),
            resnet_block(*block_arch[2]),
            resnet_block(*block_arch[3]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, X):
        return self.net(X)
