import torch
import torchvision
import argparse
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
from model import AlexNet


def evaluate(model, data_iter):
    """ 在测试集上评估泛化性 """
    with torch.no_grad():
        correct, total = 0, 0
        for X, y in data_iter:
            X, y = X.to(next(iter(model.parameters())).device), y.to(next(iter(model.parameters())).device)
            y_hat = model(X)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()
        acc = correct / total
        print(f'evaluate accuracy {acc * 100:.2f}%')
        return acc


def main():
    """ 训练主函数 """
    parser = argparse.ArgumentParser(description='AlexNet by sc 2022.12.07')
    parser.add_argument('--batch_size', default=64, type=int, help='批量大小')
    parser.add_argument('--num_workers', default=4, type=int, help='加载数据进程数量')
    parser.add_argument('--num_epochs', default=10, type=int, help='训练周期数量')
    parser.add_argument('--lr', default=.01, type=float, help='学习率')
    parser.add_argument('--log_step', default=10, type=int, help='打印log间隔的iter数量')
    parser.add_argument('--gpu', default='0', type=str, help='指定GPU, 例如 --gpu 0|1|2')
    parser.add_argument('--logger_name', default='model_best.pth', type=str, help='训练权重保存文件名')
    parser.add_argument('--resume', default='', type=str, help='加载预训练模型/继续训练，例如 ./runs/model_best.pth')

    opt = parser.parse_args()
    print(f'hyper params:{opt}')

    # 检查是否指定GPU、检查GPU是否可用
    device = torch.device('cpu')
    if opt.gpu:
        device = torch.device(f'cuda:{opt.gpu}') if torch.cuda.device_count() else torch.device('cpu')
    print(f'training on device:{device}')

    # 构造dataloader
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize([224, 224])])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data/', train=True, transform=trans, download=True, )
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = data.DataLoader(dataset=mnist_train, batch_size=opt.batch_size, shuffle=True,
                                 num_workers=opt.num_workers)
    test_iter = data.DataLoader(dataset=mnist_test, batch_size=opt.batch_size, num_workers=opt.num_workers)

    # 构造模型
    model = AlexNet()
    model.to(device)

    # --resume
    if opt.resume:
        print(f'loadding pretrain model {opt.resume}')
        model.load_state_dict(torch.load(opt.resume))

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    loss = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(opt.num_epochs):
        correct, total = 0, 0
        for i, (X, y) in enumerate(train_iter):
            # 梯度清零、计算损失、梯度反向传播、更新参数
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            iter_correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
            iter_total = y.numel()
            if i % opt.log_step == 0:
                print(f'epoch {epoch + 1}/{opt.num_epochs} iter {i}/{len(train_iter)} loss {l:.4f} '
                      f'train accuracy:{iter_correct / iter_total * 100:.2f}%')
            correct += iter_correct
            total += iter_total
        print(f'epoch {epoch + 1}/{opt.num_epochs} train accuracy:{correct / total * 100:.2f}%')

        eval_acc = evaluate(model, test_iter)
        if eval_acc > best_acc:
            print(f'A better result, updating model.')
            os.makedirs('./runs/', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('./runs/', opt.logger_name))


if __name__ == '__main__':
    main()

    # model = AlexNet()
    # X = torch.randn(1, 1, 224, 224)
    # for layer in model.net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shpae:\t', X.shape)
