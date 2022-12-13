from torch import nn


def nin_block(in_channel, out_channel, kernel_size, stride, padding):
    block = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding), nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=1), nn.ReLU()
    )
    return block


class NiN(nn.Module):
    def __init__(self, conv_arch):
        super().__init__()
        self.net = nn.Sequential()
        for i, arch in enumerate(conv_arch):
            block = nin_block(arch[0], arch[1], arch[2], arch[3], arch[4])
            self.net.add_module(f'block_{i}', block)
            if i < 3:
                self.net.add_module(f'maxpool_{i}', nn.MaxPool2d(kernel_size=3, stride=2))
        self.net.add_module(f'GAP', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.net.add_module(f'flatten', nn.Flatten())

    def forward(self, X):
        X = self.net(X)
        return X
