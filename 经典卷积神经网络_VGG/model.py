import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layer = []
    for _ in range(num_convs):
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layer.append(nn.ReLU())
        in_channels = out_channels
    layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layer)


def vgg(conv_arch):
    vgg_net = nn.Sequential()
    in_channel = 1
    for i, (num_conv, out_channel) in enumerate(conv_arch):
        vgg_net.add_module(f'block{i}', vgg_block(num_conv, in_channel, out_channel))
        in_channel = out_channel
    vgg_net.add_module('flatten', nn.Flatten())
    vgg_net.add_module('linear1', nn.Linear(out_channel * 7 * 7, 4096))
    vgg_net.add_module('relu1', nn.ReLU())
    vgg_net.add_module('dropout1', nn.Dropout(.5))
    vgg_net.add_module('linear2', nn.Linear(4096, 4096))
    vgg_net.add_module('relu2', nn.ReLU())
    vgg_net.add_module('dropout2', nn.Dropout(.5))
    vgg_net.add_module('linear3', nn.Linear(4096, 10))
    return vgg_net

