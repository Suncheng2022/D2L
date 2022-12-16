"""
ResNet适用.5或更大学习率loss=nan，小学习率效果好
1个epoch，test精度就84%
"""
import torch
import torchvision
import argparse
import os

from model import ResModel
from torchvision import transforms
from torch.utils import data



def validate(model, test_iter):
    with torch.no_grad():
        device = next(iter(model.parameters())).device
        print(f'evaludate on {device}')
        correct, total = 0, 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()
        print(f'evaluate accuracy {correct / total * 100:.2f}%')
        return correct / total


def main():
    parser = argparse.ArgumentParser(description='ResNet by sc. 2022.12.16')
    parser.add_argument('--gpu', default=0, type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--num_workers', default=8, type=int, help='')
    parser.add_argument('--num_epochs', default=10, type=int, help='')
    parser.add_argument('--lr', default=.1, type=float, help='')
    parser.add_argument('--resize', default=96, type=int, help='')


    opt = parser.parse_args()
    print(opt)
    device = torch.device(f'cuda:{opt.gpu}') if torch.cuda.device_count() else torch.device(f'cpu')
    print(f'training on {device}')

    block_arch = [[1, 64, 2, True],
                  [64, 128, 2],
                  [128, 256, 2],
                  [256, 512, 2]]    # 可自定义
    model = ResModel(block_arch)
    model.to(device)

    # X = torch.rand(size=(1, 1, 224, 224))
    # for block in model.net:
    #     X = block(X)
    #     print(f'{block.__class__.__name__}\tout shape{X.shape}')

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(opt.resize),
    ])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = data.DataLoader(dataset=mnist_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_iter = data.DataLoader(dataset=mnist_test, batch_size=opt.batch_size, num_workers=opt.num_workers)

    lr = opt.lr
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(opt.num_epochs):
        model.train()
        correct = 0
        total = 0
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            iter_correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
            iter_total = y.numel()
            print(f'epoch {epoch + 1}/{opt.num_epochs} iter {i}/{len(train_iter)} loss {l.item()} '
                  f'accuracy {iter_correct / iter_total * 100:.2f}%')
            correct += iter_correct
            total += iter_total
        train_acc = correct / total
        print(f'epoch {epoch + 1}/{opt.num_epochs} train accuracy {train_acc * 100:.2f}%')
        os.makedirs('./runs', exist_ok=True)
        torch.save(model.state_dict(), './runs/checkpoint.pth')
        eval_acc = validate(model, test_iter)
        if eval_acc > best_acc:
            best_acc = eval_acc
            print(f'updating model...')
            torch.save(model.state_dict(), './runs/model_best.pth')


if __name__ == '__main__':
    main()
