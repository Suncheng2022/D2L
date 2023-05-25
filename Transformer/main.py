import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

# embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
# input1 = torch.LongTensor([[1, 2, 4, 5],
#                            [4, 3, 2, 9]])
# print(embedding(input1))


class Embeddings(nn.Module):
    """ 构建Embeddings类实现文本嵌入层 """
    def __init__(self, d_model, vocab):
        """
        注意参数的意思
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """ x：输入的文本通过词汇映射后的数字张量 """
        return self.lut(x) * math.sqrt(self.d_model)    # 要进行一下缩放


class PositionEmbedding(nn.Module):
    """ 构建位置编码类 """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: 词嵌入维度
        :param dropout: dropout置零比率
        :param max_len: 每个句子最大长度
        """
        super(PositionEmbedding, self).__init__()
        # 实例化dropout层
        self.dropout = nn.Dropout(dropout)
        # 初始化位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(size=(max_len, d_model))
        # 初始化绝对位置矩阵 [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义变化矩阵，跳跃式变化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.) / d_model))
        # 对pe的奇数位置、偶数位置分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 对pe升维 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 将pe注册成模型的buffer，这个buffer不是模型参数，不参与训练更新
        # 即注册buffer后，我们就可以在保存模型后，加载模型时，将模型参数和这个buffer一同加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: 文本序列的词嵌入 猜测是Embeddings类的输出，[句子数，词数(可能就是max_len)，d_model] """
        # pe默认的编码长度--即第二个维度max_len太长了，默认是5000或传参，把它缩小成句子x的长度即可
        x += Variable(self.pe[:, :x.size(1), :], requires_grad=False)   # 位置编码是不参与训练
        return self.dropout(x)


if __name__ == '__main__':
    d_model = 512
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 508],
                                   [491, 998, 1, 221]]))
    emb = Embeddings(d_model, vocab)
    output = emb(x)
    # print(output, output.shape)     # torch.Size([2, 4, 512]

    # m = nn.Dropout(.2)
    # x = torch.randn(4, 5)
    # output = m(x)
    # print(output)

    # x = torch.tensor([1, 2, 3, 4])
    # print(x, x.shape)      # torch.Size([4])
    # y = torch.unsqueeze(input=x, dim=0)
    # print(y, y.shape)
    # y = torch.unsqueeze(x, 1)
    # print(y, y.shape)

    # 测试PositionEmbedding类
    d_model = 512
    vocab = 1000
    dropout = .1
    max_len = 60

    emb_layer = Embeddings(d_model, vocab)
    pe_layer = PositionEmbedding(d_model, dropout, max_len)
    x = Variable(torch.LongTensor([[100, 2, 421, 508],
                                   [491, 998, 1, 221]]))
    x_emb = emb_layer(x)
    x_emb_pe = pe_layer(x_emb)
    print(x_emb_pe, x_emb_pe.shape)     # torch.Size([2, 4, 512]

    # 可视化位置编码信息
    # 设置画布
    plt.figure(figsize=(15, 5))
    # 实例化PositionEmbedding类，dropout设为0是为了把位置编码全显示出来
    pe_layer = PositionEmbedding(d_model=20, dropout=0.)
    # 向pe中传入全零向量，为的是只显示编码位置信息
    y = pe_layer(Variable(torch.zeros(1, 100, 20)))     # 本应传入词嵌入结果
    # 画数据
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()      # 奇数开头的从同一起点开始，意为使用的同一sin/cos编码；偶数开头的同

