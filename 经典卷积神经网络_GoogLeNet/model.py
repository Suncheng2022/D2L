import torch

from torch import nn


class Inception(nn.Module):
    """ 构建Inception块 """
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1), nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU()
        )

    def forward(self, X):
        x1 = self.p1(X)
        x2 = self.p2(X)
        x3 = self.p3(X)
        x4 = self.p4(X)
        return torch.cat((x1, x2, x3, x4), dim=1)


def build_inception_block(block_arch):
    """ 构建含有多个 Inception块 的InceptionBlock结构——即多个Inception块叠加 """
    block = nn.Sequential()
    for i, inception_arch in enumerate(block_arch):
        block.add_module(f'Inception_{i}', Inception(*inception_arch))
    return block


class GoogLeNet(nn.Module):
    def __init__(self, three_block_arch):
        """
        只需提供后面3个stage的Inception块参数，其余已写死
        param three_block_arch: 包含Inception块的3个block参数
        """
        super().__init__()
        # build block one--reduced by a factor of 4
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),    # 高宽刚好减半。应该是padding和kernel_size刚好抵消，只stride起到控制减半作用了
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # 高宽刚好减半
        )
        # build block two--reduced by a factor of 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # build block three--reduced by a factor of 2
        self.block_3 = build_inception_block(three_block_arch[0])
        self.block_3.add_module(f'block_3 maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # build block four--reduced by a factor of 4
        self.block_4 = build_inception_block(three_block_arch[1])
        self.block_4.add_module(f'block_4 maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # build block five
        self.block_5 = build_inception_block(three_block_arch[2])
        self.block_5.add_module(f'block_5 GAP', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.block_5.add_module('dropout', nn.Dropout(.5))
        self.block_5.add_module(f'block_5 flatten', nn.Flatten())

        self.net = nn.Sequential(
            self.block_1, self.block_2, self.block_3, self.block_4, self.block_5,
            nn.Linear(1024, 10), nn.Softmax(dim=1)
        )
        print(f'Constructing GoogLeNet successfully!')

    def forward(self, X):
        return self.net(X)
