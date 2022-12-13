"""
2022.12.07
https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html
AlexNet分类模型
TO-DO:
    AlexNet有一层conv输出通道也是384，尝试一下
    初始化
"""
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),   # output:[96,54,54]
            nn.MaxPool2d(kernel_size=3, stride=2),                              # output:[96,26,26]
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),            # output:[256,26,26]
            nn.MaxPool2d(kernel_size=3, stride=2),                              # output:[256,12,12]
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),           # output:[384,12,12]
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),           # output:[384,12,12]
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),           # output:[256,12,12]    AlexNet实现输出也是384
            nn.MaxPool2d(kernel_size=3, stride=2),                              # output:[256,5,5]
            nn.Flatten(),                                                       # output:[6400]
            nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),                            # output:[4096]
            nn.Dropout(p=.5),
            nn.Linear(4096, 4096), nn.ReLU(),                                   # output:[4096]
            nn.Dropout(p=.5),
            nn.Linear(4096, 10)                                                 # output:[10]
        )

    def forward(self, X):
        return self.net(X)
