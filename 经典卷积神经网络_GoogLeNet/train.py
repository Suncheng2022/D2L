"""
遇到不收敛问题，表现与NiN时类似，loss徘徊在2.3附近——dropout可以在一定程度上缓解，但训练仍有波动
后lr=.1训练一次46%, lr=.3微调一次-->81.5%
"""
import argparse
import os

import torch
import torchvision

from torch import nn
from torch.utils import data
from torchvision import transforms
from model import GoogLeNet


def validate(model, test_iter):
    with torch.no_grad():
        device = next(iter(model.parameters())).device
        correct = 0
        total = 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()
        return correct / total


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        # print(f'------>init weights<------')
        nn.init.xavier_uniform_(m.weight)


def main():
    parser = argparse.ArgumentParser(description='GoogLeNet by sc. 2022.12.13')
    parser.add_argument('--gpu', default=0, type=str, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--num_workers', default=8, type=int, help='')
    parser.add_argument('--num_epochs', default=10, type=int, help='')
    parser.add_argument('--lr', default=.3, type=float, help='')
    parser.add_argument('--resume', default='', type=str, help='')
    parser.add_argument('--logger_step', default=10, type=int, help='')
    parser.add_argument('--val', default='', action='store_true', help='just validate model.')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}') if torch.cuda.device_count() else torch.device(f'cpu')
    # Construct GoogLeNet Model
    three_block_arch = [[[192, 64, (96, 128), (16, 32), 32],
                         [256, 128, (128, 192), (32, 96), 64]],

                        [[480, 192, (96, 208), (16, 48), 64],
                         [512, 160, (112, 224), (24, 64), 64],
                         [512, 128, (128, 256), (24, 64), 64],
                         [512, 112, (144, 288), (32, 64), 64],
                         [528, 256, (160, 320), (32, 128), 128]],

                        [[832, 256, (160, 320), (32, 128), 128],
                         [832, 384, (192, 384), (48, 128), 128]]]
    model = GoogLeNet(three_block_arch)
    model.to(device)
    print(f'training on {device}.')

    # initialize model
    if opt.resume:
        print(f'loading pretrain model')
        model.load_state_dict(torch.load(opt.resume))
    else:
        print(f'xavier init')
        model.apply(init_weights)

    # Testing input and output
    # X = torch.rand(size=(1, 1, 224, 224))
    # for block in model.net:
    #     X = block(X)
    #     print(f'{block.__class__.__name__} output shape \t{X.shape}')

    # construct dataloader
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(96)
    ])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = data.DataLoader(dataset=mnist_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_iter = data.DataLoader(dataset=mnist_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    if opt.val:
        print(f'current model validate accuracy {validate(model, test_iter) * 100:.2f}%')
        return

    lr = opt.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(opt.num_epochs):
        model.train()
        correct, total = 0, 0
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            iter_correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
            iter_total = y.numel()
            if i % opt.logger_step == 0:
                print(f'{epoch + 1}/{opt.num_epochs} iter {i}/{len(train_iter)} loss {l.item()} '
                      f'accuracy {iter_correct / iter_total * 100:.2f}%')
            correct += iter_correct
            total += iter_total
        print(f'{epoch + 1}/{opt.num_epochs} train accuracy {correct / total * 100:.2f}%')
        val_acc = validate(model, test_iter)
        if val_acc > best_acc:
            print(f'better model, update model. evaluate accuracy {val_acc * 100:.2f}%')
            best_acc = val_acc
            os.makedirs('./runs', exist_ok=True)
            torch.save(model.state_dict(), './runs/model_best.pth')


if __name__ == '__main__':
    main()
