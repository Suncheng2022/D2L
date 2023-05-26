import torch
import torchvision
import argparse
import os

from torch import nn
from torch.utils import data
from torchvision import transforms
from model import vgg


def evaluate(model, test_iter):
    with torch.no_grad():
        device = next(iter(model.parameters())).device
        correct, total = 0, 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()
        eval_acc = correct / total
        print(f'evaluate accuracy {eval_acc * 100:.2f}%')
        return eval_acc


def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_uniform_(m.weight)


def main():
    parser = argparse.ArgumentParser(description='Vgg by sc. 2022.12.08')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--num_workers', default=4, type=int, help='')
    parser.add_argument('--lr', default=.05, type=float, help='')
    parser.add_argument('--num_epochs', default=10, type=int, help='')
    parser.add_argument('--gpu', default=0, type=str, help='')
    parser.add_argument('--log_step', default=10, type=int, help='')
    parser.add_argument('--val', action='store_true', help='Just evaluate model.')
    parser.add_argument('--init_weights', action='store_true', help='Using Xavier init weights.')
    parser.add_argument('--resume', default='', metavar='PATH', help='')

    opt = parser.parse_args()
    print(f'Hyper Params:{opt}')

    # device = torch.device(f'cuda:0') if torch.cuda.device_count() else torch.device(f'cpu')
    device = torch.device(f'mps')
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # 共8个卷积层--再加上3个全连接层就是VGG11
    small_conv_arch = ((pair[0], pair[1] // 4) for pair in conv_arch)
    model = vgg(small_conv_arch)
    model.to(device)

    assert (opt.resume and opt.init_weights) or (opt.init_weights and not opt.resume), '--resume --init_weights conflict!'
    # 仅 --resume
    if opt.resume and not opt.init_weights:
        print(f'Loading pretrain model...')
        model.load_state_dict(torch.load('./runs/model_best.pth'))

    # 仅 --init_weights
    if opt.init_weights and not opt.resume:
        print(f'Initing weights, not stochastic...')
        model.apply(init_weights)

    # 测试输入输出
    # X = torch.randn(size=(1, 1, 224, 224))
    # for blk in model:
    #     X = blk(X)
    #     print(f'{blk.__class__.__name__} output shape\t{X.shape}')

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224])
    ])
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = data.DataLoader(mnist_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size=opt.batch_size, num_workers=opt.num_workers)

    if opt.val:
        model.load_state_dict(torch.load('./runs/model_best.pth'))
        _ = evaluate(model, test_iter)
        return

    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(opt.num_epochs):
        correct = 0
        total = 0
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            y_hat = torch.softmax(y_hat, dim=1)
            iter_correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
            iter_total = y.numel()
            if i % opt.log_step == 0:
                print(f'epoch {epoch + 1}/{opt.num_epochs} iter {i}/{len(train_iter)} '
                      f'acc {iter_correct / iter_total * 100:.2f}% loss:{l:.4f}')
            correct += iter_correct
            total += iter_total
        epoch_acc = correct / total
        print(f'epoch {epoch + 1}/{opt.num_epochs} accuracy {epoch_acc * 100:.2f}%')

        eval_acc = evaluate(model, test_iter)
        if eval_acc > best_acc:
            print(f'better model, updating model...')
            best_acc = eval_acc
            os.makedirs('./runs', exist_ok=True)
            torch.save(model.state_dict(), './runs/model_best.pth')


if __name__ == '__main__':
    main()
