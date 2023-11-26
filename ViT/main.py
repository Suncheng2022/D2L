"""
从main.ipynb复制，转为.py脚本
问题：
    1.batch_acc batch_f1竟然一样
"""
import os
os.environ['TORCH_HOME'] = 'weights'

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import torchmetrics     # 用于精度指标计算
# 用于jupyter实时绘图
import pylab as pl
from IPython import display
# 用于.py中绘图
import matplotlib.pyplot as plt

from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser(description='ViT Classification Task. scc. 2023.11.25')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--batch', default=64, type=int, help='')
    parser.add_argument('--num_workers', default=4, type=int, help='')
    parser.add_argument('--drop_last', default=True, type=bool, help='')
    parser.add_argument('--lr', default=0.005, type=float, help='')
    parser.add_argument('--save_dir', default='/Users/sunchengcheng/Projects/D2L/ViT/weights', type=str, help='')
    parser.add_argument('--log_interval', default=1, type=int, help='log打印间隔多少iters')
    return parser.parse_args()


def get_dataloader(opt):
    """ 构建数据集，返回dataloader """
    process_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),        # int 图片短边会缩放到int，长边相应缩放，不是我们想要的正方形
    ])
    # CIFAR10 32x32 colour images in 10 classes
    train_dataset = torchvision.datasets.CIFAR10(root='../data/', train=True, transform=process_data, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../data/', transform=process_data, download=True)
    train_iter = DataLoader(dataset=train_dataset, batch_size=opt.batch, shuffle=True, num_workers=opt.num_workers, drop_last=opt.drop_last)
    test_iter = DataLoader(dataset=test_dataset, batch_size=opt.batch, shuffle=False, num_workers=opt.num_workers, drop_last=opt.drop_last)
    return train_iter, test_iter


def evaluate(model, test_iter, device):
    metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=len(test_iter.dataset.classes)).to(device)
    """ 使用验证集评估模型 """
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(tqdm(test_iter)):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            correct += torch.sum(y_hat.softmax(dim=-1).argmax(dim=-1) == y)
            total += x.shape[0]
            acc = correct / total
            batch_f1 = metric_f1(y_hat.softmax(dim=-1).argmax(dim=-1), y)
            # print(f'---------->evaluate {i}/{len(test_iter)} acc:{acc.item() * 100:.7f}% batch_f1 {batch_f1 * 100:.4f}% ')
        eval_f1 = metric_f1.compute()
        print(f'---------->evaluate acc:{acc.item() * 100:.7f}% eval_f1:{eval_f1 * 100:.4f}%')
        return acc.item(), eval_f1


def main():
    opt = parser()
    train_iter, test_iter = get_dataloader(opt)
    """ 定义ViT_model """
    ViT_model = torchvision.models.vit_b_16(num_classes=len(train_iter.dataset.classes))
    weights = torch.load('weights/hub/checkpoints/vit_b_16-c867db91.pth')
    del weights['heads.head.weight']
    del weights['heads.head.bias']
    ViT_model.load_state_dict(weights, strict=False)
    # print(weights.keys())
    # exit(-1)

    # 测试vgg模型，后期注释
    # ViT_model = torchvision.models.vgg11(num_classes=len(train_iter.dataset.classes))
    # weights_vgg = torch.load(f='weights/hub/checkpoints/vgg11-8a719046.pth')
    # del weights_vgg['classifier.6.weight']
    # del weights_vgg['classifier.6.bias']
    # ViT_model.load_state_dict(weights_vgg, strict=False)

    # 废弃
    # ViT_model.heads = nn.Linear(768, train_dataset.classes.__len__())
    # """ 测试ViT输入输出 """
    # with torch.no_grad():
    #     ViT_model.eval()        # 影响BN、dropout层
    #     x = torch.rand(size=(10, 3, 224, 224))
    #     y = ViT_model(x)
    #     print(y.softmax(dim=-1), torch.argmax(torch.softmax(y, dim=-1), dim=-1), sep='\n')

    """ 定义loss、optim """
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=ViT_model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=train_iter.__len__())

    """ 训练 """
    max_f1 = 0      # 记录最大f1分数，保存模型
    lr_iter = []
    acc_iter = []
    batch_acc_iter = []
    batch_f1_iter = []
    correct = 0
    total = 0
    device = torch.device('mps')
    # 精度指标计算
    metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(train_iter.dataset.classes), average='macro').to(device)
    metric_precision = torchmetrics.Precision(task='multiclass', num_classes=len(train_iter.dataset.classes), average='macro').to(device)
    metric_recall = torchmetrics.Recall(task='multiclass', num_classes=len(train_iter.dataset.classes), average='macro').to(device)
    metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=len(train_iter.dataset.classes), average='macro').to(device)
    # metric_precision = torchmetrics.Precision
    ViT_model.to(device=device)
    for epoch in range(opt.epochs):
        ViT_model.train()               # 切换到train模式
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = ViT_model(x)
            # 统计精度
            correct += torch.sum(y_hat.softmax(dim=-1).argmax(dim=-1) == y)
            total += x.shape[0]
            acc = correct / total
            acc_iter.append(acc.item())
            batch_acc = metric_acc(y_hat.softmax(dim=-1).argmax(dim=-1), y)     # 计算batch的精度；已验证metric_acc.compute()与自己写的全局acc同
            batch_precision = metric_precision(y_hat.softmax(dim=-1).argmax(dim=-1), y)
            batch_recall = metric_recall(y_hat.softmax(dim=-1).argmax(dim=-1), y)
            batch_f1 = metric_f1(y_hat.softmax(dim=-1).argmax(dim=-1), y)
            lr_iter.append(scheduler.get_last_lr()[0])
            batch_acc_iter.append(batch_acc.item())
            batch_f1_iter.append(batch_f1.item())
            if i % opt.log_interval == 0:
                print(f'epoch {epoch}/{opt.epochs} iter {i}/{len(train_iter)} lr {lr_iter[-1]:.7f} '
                      f'acc {acc * 100:.4f}% batch_acc {batch_acc * 100:.4f}% '
                      f'batch_precision {batch_precision * 100:.4f} '
                      f'batch_recall {batch_recall * 100:.4f} f1_ {2 * batch_precision * batch_recall / (batch_precision + batch_recall) * 100:.4f}% '
                      f'batch_f1 {batch_f1 * 100:.4f}% ')
            # backward update
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # draw in jupyter
            # pl.clf()
            # pl.plot(acc_iter, label='acc')
            # pl.plot(lr_iter, label='lr * 200')
            # pl.plot(batch_acc_iter, label='batch_acc')
            # pl.plot(batch_f1_iter, label='batch_f1')
            # pl.legend(loc='upper right')                # 必须设置，否则pl.plot()的label参数显示不出来
            # pl.xlabel(f'iters')
            # display.display(pl.gcf())
            # display.clear_output(True)
            # draw in .py
            plt.plot(acc_iter, label='acc')
            plt.plot([i * 200 for i in lr_iter], label='lr * 200')
            plt.plot(batch_acc_iter, label='batch_acc')
            plt.plot(batch_f1_iter, label='batch_f1')
            plt.legend(loc='upper right')                # 必须设置，否则pl.plot()的label参数显示不出来
            plt.xlabel(f'iters')
            plt.savefig(os.path.join(opt.save_dir, f'lr_acc_f1.png'))
            plt.clf()
        # evalute model
        epoch_acc = metric_acc.compute()
        epoch_precision = metric_precision.compute()
        epoch_recall = metric_recall.compute()
        epoch_f1 = metric_f1.compute()
        metric_acc.reset()
        metric_precision.reset()
        metric_recall.reset()
        metric_f1.reset()
        print(f'epoch {epoch}/{opt.epochs} epoch performance: epoch_acc {epoch_acc * 100:.4f} epoch_precision {epoch_precision * 100:.4f} epoch_recall {epoch_recall * 100:.4f} epoch_f1 {epoch_f1 * 100:.4f}')
        acc, eval_f1 = evaluate(ViT_model, test_iter, device)
        if eval_f1 > max_f1:
            max_f1 = eval_f1
            print(f'*************** Find Better Model, Saving Model to {opt.save_dir} *****************')
            torch.save(ViT_model.state_dict(), os.path.join(opt.save_dir, f'best.pth'))


if __name__ == '__main__':
    main()