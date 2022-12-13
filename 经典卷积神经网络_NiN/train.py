import argparse
import torchvision
import torch

from model import NiN
from torchvision import transforms
from torch.utils import data
from torch import nn

def main():
    parser = argparse.ArgumentParser(description='Nin by sc. 2022.12.12')
    parser.add_argument('--batch_size', default=64, type=int, help='')
    parser.add_argument('--num_workers', default=4, type=int, help='')
    parser.add_argument('--lr', default=.1, type=float, help='')
    parser.add_argument('--logger_step', default=100, type=int, help='')
    parser.add_argument('--num_epochs', default=10, type=int, help='')

    opt = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.device_count() else torch.device('cpu')
    print(f'training on {device}')

    # 构建NiN网络模型
    conv_arch = [[1, 96, 11, 4, 0], [96, 256, 5, 1, 2], [256, 384, 3, 1, 1], [384, 10, 3, 1, 1]]
    model = NiN(conv_arch)
    model.to(device)

    # 测试模型输入输出
    # X = torch.rand(size=(1, 1, 224, 224))
    # for layer in model.net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'out shape\t', X.shape)

    # 构造date-loader
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224)
    ])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = data.DataLoader(dataset=mnist_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_iter = data.DataLoader(dataset=mnist_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 训练过程
    lr = opt.lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(opt.num_epochs):
        correct, total = 0, 0
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            iter_correct, iter_total = 0, 0
            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            iter_correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
            iter_total = y.numel()
            correct += iter_correct
            total += iter_total
            if i % opt.logger_step == 0:
                print(f'{epoch}/{opt.num_epochs} iter {i}/{len(train_iter)} loss {l.item():.4f} accuracy {iter_correct / iter_total * 100:.2f}%')
        print(f'{epoch}/{opt.num_epochs} accuracy {correct / total * 100:.2f}%')


if __name__ == '__main__':
    main()
