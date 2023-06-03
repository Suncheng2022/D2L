""" 教程里新建.py
Field的导入报错 尝试torchtext==0.6版本成功
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入经典文本数据集工具包
import torchtext
# 导入英文分词工具包
from torchtext.data.utils import get_tokenizer
# 导入已经构建完成的Transformer工具包
from pyitcast.transformer import TransformerModel

import os

os.environ['TORCH_HOME'] = './data'

# 将数据进行语料库封装
TEXT = torchtext.data.Field(tokenize=get_tokenizer('basic_english'),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
# print(TEXT)   # 返回迭代器

# 使用torchtext的数据集方法导入WikiText2数据集
train_text, val_text, test_text = torchtext.datasets.WikiText2.splits(text_field=TEXT, root='data')
# print(test_text.examples[0].text[:100])

# 将训练集文本数据构建一个vocab对象，可以使用vocab对象的stoi方法共包含的不重复的词汇总数
TEXT.build_vocab(train_text)

# 设置GPU
# device = torch.device('mps' if torch.cuda.is_available() else 'cpu')    # NVIDIA GPU改为 'cuda'; Apple M芯片改为 'mps'
device = torch.device('mps')

# 构建批次数据的函数
def batchify(data, batch_size):
    """
    :param data: 之前得到的文本数据 train_text val_text test_text; 使用data.examples[0].text[:10]查看字符
    :param batch_size:
    """
    # 第一步使用TEXT的numericalize()将单词映射成连续数字
    data = TEXT.numericalize([data.examples[0].text])   # data.examples.__len__()为1 这句是将data的所有单词映射为数字

    # 第二步取得需要经过多少次的batch_size后能够遍历完所有数据
    nbatch = data.size(0) // batch_size     # data.size() 训练集torch.Size([2086708, 1])

    # 利用narrow()对数据进行切割
    # 第1个参数 代表横轴切割还是纵轴切割  0-横轴 1-纵轴
    # 第2个参数 第3个参数分别代表切割的起始位置和终止位置
    data = data.narrow(0, 0, nbatch * batch_size)  # 使用的数据刚好是 整数 个batch_size

    # 对data形状进行转变
    data = data.view(batch_size, -1).t().contiguous()  # t()让batch_size到列上 [nbatch, batch_size]
    return data.to(device)


# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(x.narrow(0, 0, 2))    # 行 左闭右开
# print(x.narrow(1, 1, 2))    # 列 闭区间

# 首先设置训练数据批次大小
batch_size = 20
# 设置验证数据和测试数据批次大小
eval_batch_size = 10

# 获得训练数据、验证数据、测试数据
train_data = batchify(train_text, batch_size)   # [nbatch, batch_size]
val_data = batchify(val_text, eval_batch_size)
test_data = batchify(test_text, eval_batch_size)

# 设定句子最大长度
bptt = 35


def get_batch(source, i):
    """
    获取batch行文本
    :param source:  train_data等
    :param i: 批次数
    :return: data, target
    """
    # 确定句子长度值
    seq_len = min(bptt, len(source) - 1 - i)

    # 首先得到源数据
    data = source[i:i + seq_len]
    # 然后得到目标数据
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


# source = test_data
# i = 1
# x, y = get_batch(source, i)
# print(x, y, sep='\n')

# 设置模型超参数
# 通过TEXT.vocab.stoi方法获取不重复的词汇总数
ntokens = len(TEXT.vocab.stoi)      # 28785

# 设置词嵌入维度的值等于200
emsize = 200

# 设置前馈全连接层节点数
nhid = 200

# 设置编码器层的层数
nlayers = 2

# 多头注意力机制头数
nhead = 2

# 设置置零比率
dropout = .2

# 将参数传入TransformerModel实例化模型
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# loss
criterion = nn.CrossEntropyLoss()

# lr
lr = 5.

# optim
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 定义lr调整器，使用torch自带的lr_scheduler，将优化器传入
schedualer = torch.optim.lr_scheduler.StepLR(optimizer, 1., gamma=.95)

import time


# 构建训练函数
def train():
    # 首先开启train模式
    model.train()
    # 定义初始loss值
    total_loss = 0
    # 设置打印间隔
    log_interval = 200
    # 获取当前开始时间
    start_time = time.time()
    # 遍历训练数据，训练模型
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):  # train_data.size() [104335, 20]
        # 通过前面的get_batch获取源数据和目标数据
        data, targets = get_batch(train_data, i)    # data [bptt, batch] target形状[bptt * batch]
        # 梯度归零
        optimizer.zero_grad()
        # 通过模型预测输出  维度ntokens即对每个词预测的概率
        output = model(data)    # torch.Size([bbpt35, batch20, ntokens28785])
        # 计算loss    报错
        loss = criterion(output.view(-1, ntokens), targets)     # output.view()后 [bbpt * batch, ntokens] target [bbpt * batch]
        # 反传
        loss.backward()
        # 剪裁梯度，防止梯度爆炸、消失
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        # 参数更新
        optimizer.step()
        # 累加损失值
        total_loss += loss.item()
        # 打印日志
        if batch % log_interval == 0 and batch > 0:
            # 首先计算平均损失
            cur_loss = total_loss / log_interval
            # 计算训练到目前的耗时
            elapased = time.time() - start_time
            # 打印
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // bptt, schedualer.get_lr()[0],
                                                      elapased * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            # 每个打印批次结束后，将总损失清零
            total_loss = 0
            # 重新获取下一个打印轮次的开始时间
            start_time = time.time()


# 构建评估函数
def evaluate(eval_model, data_source):
    """
    :param eval_model: 训练后的模型
    :param data_source: 验证集、测试集数据
    :return:
    """
    eval_model.eval()

    # 初始化总损失
    total_loss = 0
    # 模型开启评估模式后，不进行反传，以加快计算
    with torch.no_grad():
        # 遍历验证数据
        for i in range(0, data_source.size(0) - 1, bptt):
            # 首先通过get_batch()获取源数据和目标数据
            data, targets = get_batch(data_source, i)
            # 将源数据放入评估模型中，进行预测
            output = eval_model(data)
            # 对输出张量进行变形，遍历全部词汇的概率分布
            output_flat = output.view(-1, ntokens)      # [拉平，总共有多少单词] 每一个单词都有一个概率
            # 累加损失
            total_loss += criterion(output_flat, targets).item()
    # 返回评估的总损失值
    return total_loss


# 首先初始化最佳模型损失值
best_val_loss = float('inf')

epochs = 3

# 定义最佳模型，初始化为空
best_model = None

# 训练
for epoch in range(1, epochs + 1):
    # 获取当前轮次开始时间
    start_time = time.time()
    # 直接调用训练函数进行模型训练
    train()
    # 调用评估函数得到验证集损失
    val_loss = evaluate(model, val_data)
    # 打印log
    print('-' * 90)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - start_time),
                                                                                 val_loss))
    print('-' * 90)
    # 通过比较当前epoch的损失，获取最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    # 每个epoch后调整优化器学习率
    schedualer.step()


# 添加测试流程代码
test_loss = evaluate(best_model, test_data)
print('-' * 90)
print('| End of traning | test loss {:5.2f}'.format(test_loss))
print('-' * 90)