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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 构建批次数据的函数
def batchify(data, batch_size):
    """
    :param data: 之前得到的文本数据 train_text val_text test_text
    :param batch_size:
    """
    # 第一步使用TEXT的numericalize()将单词映射成连续数字
    data = TEXT.numericalize([data.examples[0].text])

    # 第二步取得需要经过多少次的batch_size后能够遍历完所有数据
    nbatch = data.size(0) // batch_size

    # 利用narrow()对数据进行切割
    # 第1个参数 代表横轴切割还是纵轴切割  0-横轴 1-纵轴
    # 第2个参数 第3个参数分别代表切割的起始位置和终止位置
    data = data.narrow(0, 0, nbatch * batch_size)  # 使用的数据刚好是 整数 个batch_size

    # 对data形状进行转变
    data = data.view(batch_size, -1).t().contiguous()  # t()让batch_size到列上
    return data.to(device)


# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(x.narrow(0, 0, 2))    # 行 左闭右开
# print(x.narrow(1, 1, 2))    # 列 闭区间

# 首先设置训练数据批次大小
batch_size = 20
# 设置验证数据和测试数据批次大小
eval_batch_size = 10

# 获得训练数据、验证数据、测试数据
train_data = batchify(train_text, batch_size)
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
    target = source[i + 1:i + 1 + seq_len]
    return data, target


# source = test_data
# i = 1
# x, y = get_batch(source, i)
# print(x, y, sep='\n')

# 设置模型超参数
# 通过TEXT.vocab.stoi方法获取不重复的词汇总数
ntokens = len(TEXT.vocab.stoi)

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
model = TransformerModel(ntokens, emsize, nhead, nlayers, dropout).to(device)

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
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 通过前面的get_batch获取源数据和目标数据
        data, targets = get_batch(train_data, i)
        # 梯度归零
        optimizer.zero_grad()
        # 通过模型预测输出
        output = model(data)
        # 计算loss
        loss = criterion(output.view(-1, ntokens), targets)
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
                  'loss {:5.2f} | ppl {:8.2f}'.format(0, batch, len(train_data) // bptt, schedualer.get_lr(),
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
    # 模型开启评估模式后，不进行反传
    with torch.no_grad():
        # 遍历验证数据
        for i in range(0, data_source.size()  - 1,bptt);