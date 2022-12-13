"""
训练、测试、推断都实现在一个文件了，纯为测试。
TO-DO:

"""
import torch
import torchvision
import argparse
import os

from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils import data


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 1, 28, 28)  # b, c, h, w


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度:正确预测的数量，总预测的数量"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)  # [256, 10]
            # print(f'$$$$$$$ evaluate_accuracy_gpu:\n{X.shape} \n{y} \n{net(X)}')    # torch.Size([256, 1, 28, 28]) torch.Size([256]) torch.Size([256, 10])
            # print(f'$$$$$$$ argmax:{torch.argmax(y_hat, dim=1)}\ny:{y}')
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()
    return correct / total


def train_ch6(net, train_iter, test_iter, num_epochs, lr, log_step, best_acc, opt, device):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if opt.resume:
        net.load_state_dict(torch.load(opt.resume))
        print(f'加载--resume {opt.resume}')
        acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'验证精度:{acc}')
    else:
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    best_acc = best_acc
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        net.train()
        correct = 0
        total = 0
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            # 计算训练精度
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()

            if i % log_step == 0:
                print(f'epoch {epoch + 1}/{num_epochs} iter {i}/{len(train_iter)} loss {l:.4f}')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1} train accuracy {correct / total * 100:.2f}% test accuracy {test_acc * 100:.2f}%')

        if test_acc > best_acc:  # 有更好的结果，更新权重
            best_acc = test_acc
            os.makedirs('./runs', exist_ok=True)
            torch.save(net.state_dict(), './runs/model_best.pth.tar')
            print(f'更新权重:\n{net.state_dict().keys()}')


def main():
    parser = argparse.ArgumentParser(description='LetNet by sc 2022.12.06')
    parser.add_argument('--epochs', default=10, type=int, help='训练周期')
    parser.add_argument('--lr', default=.9, type=float, help='学习率')
    parser.add_argument('--batch_size', default=256, type=int, help='批量大小')
    parser.add_argument('--num_workers', default=4, type=int, help='加载批量数据的进程数量')
    parser.add_argument('--log_step', default=50, type=int, help='打印log的iter间隔')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='加载预训练模型/继续训练')

    parser.add_argument('--validate', action='store_true', help='仅做一次验证即停止程序')
    parser.add_argument('--detect', action='store_true', help='识别输入结果')
    opt = parser.parse_args()
    print(f'模型超参:{opt}')

    net = nn.Sequential(
        Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10)
    )

    if opt.detect:
        net.load_state_dict(torch.load('./runs/model_best.pth.tar'))
        img = Image.open('./images/img_2.png').convert('L')
        img = transforms.Resize([28, 28])(img)
        img = transforms.ToTensor()(img)  # 3, 28, 28
        img = net(img)
        res = torch.argmax(torch.softmax(img, dim=1), dim=1)
        fashion_mnist = {0: 'T恤', 1: '裤子', 2: '套衫', 3: '裙子', 4: '外套',
                         5: '凉鞋', 6: '汗衫', 7: '运动鞋', 8: '包', 9: '靴子'}
        print(f'识别结果:{fashion_mnist[res.item()]}')
        return

    batch_size = opt.batch_size
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data/', train=True, transform=transforms.ToTensor(), download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data/', train=False, transform=transforms.ToTensor(), download=False
    )
    train_iter = data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    acc = 0
    if opt.validate:
        print(f'仅验证.')
        net.load_state_dict(torch.load('runs/model_best.pth.tar'))
        acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'验证精度:{acc * 100:.2f}%')
        return

    lr, num_epochs = opt.lr, opt.epochs
    train_ch6(net, train_iter, test_iter, num_epochs, lr, opt.log_step, acc, opt,
              device=torch.device('cuda:0') if torch.cuda.device_count() else torch.device('cpu'))


if __name__ == '__main__':
    main()
