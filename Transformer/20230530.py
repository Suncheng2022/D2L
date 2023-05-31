""" 完全按照教程写代码 """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import pdb

from torch.autograd import Variable


class Embeddings(nn.Module):
    """ 构建Embedding类实现文本嵌入层 """
    def __init__(self, d_model, vocab):
        """
        d_model: 词嵌入维度
        vocab: 词表大小
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """ x代表输入进模型的文本通过词汇映射后的张量 """
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 421, 508],
                               [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
# print(f'embr:{embr}\n{embr.shape}')

# m = nn.Dropout(p=.2)
# input1 = torch.randn(4, 5)
# output = m(input1)
# # print(output)
#
# x = torch.tensor([1, 2, 3, 4])
# y = torch.unsqueeze(x, 0)
# print(y)
# z = torch.unsqueeze(x, 1)
# print(z)


class PositionalEncoding(nn.Module):
    """ 构建位置编码器类 """
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model: 词嵌入维度
        # dropout: 置零比率
        # max_len：每个句子最大长度
        super(PositionalEncoding, self).__init__()

        # 实例化dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵, 大小是max_len * d_model  有多少个词 * 每个词多大
        pe = torch.zeros(max_len, d_model)

        # 初始化绝对位置矩阵     max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变换矩阵div_term，跳跃式的初始化  形状 [d_model / 2] 因为要分别初始化pe的 偶数索引 和 奇数索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.) / d_model))

        # 将前面定义的变换矩阵div_term，分别用于pe的偶数位置、奇数位置的赋值
        pe[:, 0::2] = torch.sin(position * div_term)    # [max_len, 1] * [d_model / 2] --> [max_len=60, d_model / 2=256] 应该自动调用了广播机制
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量  60 * 512 --> 1 * 60 * 512
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册成模型的buffer，这个buffer不是模型中的参数，不跟随优化器同步更新
        # 注册成buffer后，我们就可以在模型保存后重新加载时，将这个位置编码器和模型参数加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: 代表文本序列的词嵌入表示   [batch 2, 4, 512]
        # 首先明确pe的编码太长了，将第二个维度也就是max_len对应的维度缩小成x的句子长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)   # requires_grad=False 代表位置编码不参与未来梯度更新
        return self.dropout(x)


d_model = 512
dropout = .1
max_len = 60

x = embr    # [2, 4, 512]
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)       # [2, 4, 512]
# print(f'{pe_result}\n{pe_result.shape}')

# 第一步 设置一个画布
# plt.figure(figsize=(15, 5))
#
# # 第二步 实例化PositionalEncoding对象，词嵌入维度20，置零比率0
# pe = PositionalEncoding(d_model=20, dropout=0)
#
# # 向pe中传入全0初始化的x，相当于展示pe
# y = pe(Variable(torch.zeros(1, 100, 20)))
#
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
#
# plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])
# plt.show()

# print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k=-1))
# print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k=0))
# print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k=1))

def subsequent_mask(size):
    """ 构建掩码张量的函数 返回下三角矩阵 [1, size, size]
        [size, size]维度，把每列视为一个进行遮挡的张量，我们得到的不是下三角矩阵嘛，也就是下半部是1，上半部分是0，教程说1表示遮挡 """
    # size: 掩码张量后两个维度，形成一个方阵
    attn_shape = (1, size, size)

    # 使用np.ones()先构建全1张量，利用np.triu()形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 使得三角矩阵反转
    return torch.from_numpy(1 - subsequent_mask)

# size = 5
# sm = subsequent_mask(size)      # [1, 5, 5]
# print(sm, sm.shape)

# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])  # 返回的是3维张量 [0]降维一下
# plt.show()

# x = Variable(torch.randn(5, 5))
# print(x)
#
# mask = Variable(torch.zeros(5, 5))
# print(mask)
#
# y = x.masked_fill(mask == 0, -1e9)
# print(y)

def attention(query, key, value, mask=None, dropout=None):
    """ q k v：代表注意力的三个输入 应该都是三维的[batch, tokens, d_model]
        mask: 掩码张量
        dropout: 实例化对象 """
    # 首先将query最后的维度提取出来，代表的是词嵌入的维度
    d_k = query.size(-1)

    # 这里是在计算权重；按照注意力计算公式，将q k转置进行matmul，除以缩放系数d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    # 是吗[batch, tokens, tokens], 是的 [2, 4, 4]
    # print(f'$$$$$$$$$$$$$$$$ {scores.shape}')

    # 判断是否使用掩码张量
    if mask is not None:
        # 利用masked_fill()，将掩码张量和0进行位置的一一比较，等于0则替换成非常小的数
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores最后一个维度进行softmax 得到对所有value注意力权重
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后一步完成p_attn和val张量的乘法，并返回query的注意力表示
    return torch.matmul(p_attn, value), p_attn      # [2, 4, 512], [2, 4, 4]


query = key = value = pe_result     # [2, 4, 512]  pe_result是经过embedding、position embedding后的结果
mask = Variable(torch.zeros(2, 4, 4))
attn, p_attn = attention(query, key, value, mask=mask)      # 传入一个全0的mask，则计算的scores = matmul(q, k.t)结果全为非常小的值了，之后softmax结果也会都一样
print(f'attn:{attn}\n{attn.shape}\np_attn:{p_attn}\n{p_attn.shape}')
