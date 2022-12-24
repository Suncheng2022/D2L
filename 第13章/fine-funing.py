"""
重新写一遍微调：
    似乎对学习率很敏感，5e-4pretrain就差、5e-5瞬间变好
    修改了初始化逻辑，if opt.fine_tune只初始化fc，否则全部初始化
    opt.fine_tune超参可以控制是否加载pretrain
    尝试了从头开始训练，resume 1次共20epochs，测试精度85.33%，也不错

    fine-fune自己的数据总是过拟合--似乎找到了原因，解决了推测时候总是出错的问题
    model.eval()模式对推断的时候影响很大！
    更换数据集后，如类别数变化要修改最后的分类全连接层
    小实验：迁移学习/微调可以将骨架网络(或者其他什么特征提取部分)学习率设为0，fc使用小学习率，效果也很好。
"""
import pdb

import torchvision
import torch
import argparse
import os

from torch import nn
from torch.utils import data
from torchvision import transforms


def init_weight(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


def evaluate(net, data_iter):
    net.eval()
    with torch.no_grad():
        device = next(iter(net.parameters())).device
        right, total = 0, 0
        for i, (X, y) in enumerate(data_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            right += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += y.numel()
        eval_acc = right / total
        print(f'evaluate accuracy {eval_acc * 100:.2f}%')
        return eval_acc


def fine_tuning_train():
    parser = argparse.ArgumentParser(description='ResNet fine-tuning by sc. 2022.12.17')
    parser.add_argument('--gpu', default=0, type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--num_epochs', default=5, type=int, help='')
    parser.add_argument('--lr', default=5e-5, type=float, help='')
    parser.add_argument('--fine_tune', action='store_true', help='load pretrain and different learning rate for different layer.')
    parser.add_argument('--logger_name', default='./runs/liushuo/model_best.pth', type=str, help='')
    parser.add_argument('--resume', default='', type=str, help='eg:./runs/model_best.pth')
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--detect', default='', help='eg ./images/001.jpg')
    parser.add_argument('--logger_step', default=10, type=int, help='')
    parser.add_argument('--num_workers', default=4, type=int, help='')

    opt = parser.parse_args()
    print(opt)
    device = torch.device(f'cuda:{opt.gpu}') if torch.cuda.device_count() > int(opt.gpu) else torch.device(f'cpu')
    print(f'running on {device}')

    # Construct the model
    model = torchvision.models.resnet18(pretrained=True if opt.fine_tune else False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)     # 根据不同数据集相应调整输出类别
    if opt.fine_tune:
        torch.nn.init.xavier_uniform_(model.fc.weight)
    else:
        model.apply(init_weight)
    model.to(device)
    if opt.resume:
        print(f'loading pretrain model {opt.resume}')
        model.load_state_dict(torch.load(opt.resume))

    # Construct dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_augs = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if opt.detect:
        from PIL import Image
        model_eval = torchvision.models.resnet18()
        model_eval.fc = torch.nn.Linear(model.fc.in_features, 3)
        model_eval.to(device)
        model_eval.load_state_dict(torch.load('./runs/liushuo/model_best.pth'))
        model_eval.eval()   # 这句话对预测结果很关键！！！

        # class_dict = {1: 'not-hotdog', 0: 'hotdog'}
        class_dict = {1: 'hotdog', 0: 'car', 2: 'liushuo'}
        # pdb.set_trace()
        img = Image.open(opt.detect)
        img = test_augs(img)
        transforms.ToPILImage()(img).show()
        img = img.unsqueeze(0)
        y_hat = model_eval(img.to(device))
        y_hat = torch.softmax(y_hat, dim=1)
        print(y_hat)
        label = torch.argmax(y_hat, dim=1)
        print(f'预测类别：{class_dict[label.item()]}')
        exit(-1)

    # data_dir = r'../data/hotdog'
    data_dir = r'../data/myperson'
    imgfld_train = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_augs)
    imgfld_test = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_augs)
    classes_dict = imgfld_train.class_to_idx
    print(f'$$$$$$$$$ {classes_dict}')
    train_iter = data.DataLoader(dataset=imgfld_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_iter = data.DataLoader(dataset=imgfld_test, batch_size=opt.batch_size, num_workers=opt.num_workers)
    # look the images after augs
    # imgs = [imgfld_train[i][0] for i in range(8)]
    # [img.show() for img in imgs]
    # exit(-1)

    # Just evaluate model's accuracy
    if opt.eval:
        _ = evaluate(model, test_iter)
        exit(-1)

    # train
    loss = torch.nn.CrossEntropyLoss(reduction='none')      # https://blog.csdn.net/goodxin_ie/article/details/89645358
    if opt.fine_tune:
        param_1x = [param for name, param in model.named_parameters() if name not in ['fc.weight', 'fc.bias']]  # use learning rate of hyperparameters.
        optimizer = torch.optim.SGD([{'params': param_1x},
                                    {'params': model.fc.weight, 'lr': opt.lr * 10}],
                                    lr=opt.lr, weight_decay=.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    best_acc = 0 if not opt.resume else evaluate(model, test_iter)
    for epoch in range(opt.num_epochs):
        model.train()
        correct = 0
        total = 0
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.sum().backward()
            optimizer.step()

            iter_correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
            iter_total = y.numel()
            if i % opt.logger_step == 0:
                print(f'epoch {epoch + 1}/{opt.num_epochs} iter {i}/{len(train_iter)} loss {l.sum():.7f} '
                      f'accuracy {iter_correct / iter_total * 100:.2f}%')
                # print(f'$$$$$$$$$$ {model.fc.weight.sum()}\t{param_1x[0].data.sum()}\t{param_1x[1].data.sum()}')
            correct += iter_correct
            total += iter_total
        train_acc = correct / total
        print(f'epoch {epoch + 1}/{opt.num_epochs} train accuracy {train_acc * 100:.2f}%')
        eval_acc = evaluate(model, test_iter)
        if eval_acc > best_acc:
            best_acc = eval_acc
            print(f'updating model.')
            os.makedirs(os.path.dirname(opt.logger_name), exist_ok=True)
            torch.save(model.state_dict(), opt.logger_name)
    print(f'{classes_dict} best_acc {best_acc * 100:.2f}%')


if __name__ == '__main__':
    fine_tuning_train()





# """
# 微调要重定义所有层的lr、对没初始化的层进行随机初始化
# """
# import os
# import torch
# import torchvision
# import pdb
#
# from torchvision import transforms
# from torch import nn
# from d2l import torch as d2l
# from PIL import Image
# from torch.utils import data
#
#
# def evaluate(net, data_iter):
#     net.eval()
#     with torch.no_grad():
#         device = next(iter(net.parameters())).device
#         right, total = 0, 0
#         for i, (X, y) in enumerate(data_iter):
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             right += torch.sum(torch.argmax(y_hat, dim=1) == y)
#             total += y.numel()
#         eval_acc = right / total
#         print(f'evaluate accuracy {eval_acc * 100:.2f}%')
#         return eval_acc
#
#
# def train_fine_tuning(net, lr=5e-4, batch_size=16, num_epochs=5, param_group=True):
#     train_iter = data.DataLoader(dataset=train_imgs, batch_size=batch_size, shuffle=True)
#     test_iter = data.DataLoader(dataset=test_imgs, batch_size=batch_size)
#     device = torch.device(f'cuda:0') if torch.cuda.device_count() else torch.device(f'cpu')
#     loss = nn.CrossEntropyLoss(reduction='none')  # reduction参数在pytorch后续版本将取代size_avg、reduce参数
#     if param_group:
#         param_1x = [param for name, param in net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
#         trainer = torch.optim.SGD([{'params': param_1x},
#                                    {'params': net.fc.parameters(), 'lr': lr * 10}],
#                                   lr=lr, weight_decay=.001)
#     else:
#         trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=.001)
#     # d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, device)
#
#     for epoch in range(num_epochs):
#         right = 0
#         total = 0
#         for i, (X, y) in enumerate(train_iter):
#             net.train()
#             trainer.zero_grad()
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             l.sum().backward()
#             trainer.step()
#
#             iter_right = torch.sum(torch.argmax(y_hat, dim=1) == y)
#             iter_total = y.numel()
#             if i % 10 == 0:
#                 print(f'{epoch + 1}/{num_epochs} iter {i}/{len(train_iter)} loss {l.sum()} '
#                       f'accuracy {iter_right / iter_total * 100:.2f}%')
#             right += iter_right
#             total += iter_total
#         train_acc = right / total
#         print(f'{epoch + 1}/{num_epochs} train accuracy {train_acc * 100:.2f}%')
#         _ = evaluate(net, test_iter)
#
#
# if __name__ == '__main__':
#     d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
#     data_dir = d2l.download_extract('hotdog')  # ..\data\hotdog
#
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     train_augs = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ])
#     test_augs = transforms.Compose([
#         transforms.Resize([256, 256]),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         normalize
#     ])
#     train_imgs = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_augs)
#     test_imgs = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_augs)
#
#     # 查看读取的正例、负例样本
#     # hotdogs = [train_imgs[i][0] for i in range(8)]
#     # [img.show() for img in hotdogs]
#     # not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
#     # [img.show() for img in not_hotdogs]
#
#     net = torchvision.models.resnet18(pretrained=True)
#     net.fc = nn.Linear(in_features=net.fc.in_features, out_features=2)
#     nn.init.xavier_uniform_(net.fc.weight)
#     net.to(torch.device(f'cuda:0'))
#     # print(f'{net.fc}')
#     train_fine_tuning(net, lr=5e-5)
