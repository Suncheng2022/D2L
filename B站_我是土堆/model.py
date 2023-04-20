import torch
from torch import nn


class MySequence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.seq(x)
        return x


if __name__ == "__main__":
    # 验证模型的正确性
    x = torch.zeros(size=(32, 3, 32, 32))
    model = MySequence()
    output = model(x)
    print(output.shape)