""" 完全按照教程写代码
2023.06.02
    修改了库代码
    位置 D:\ProgramData\miniconda3\envs\pt20\Lib\site-packages\pyitcast\transformer_utils.py line134
    默认为 loss.data[0] * norm，改为return loss.data.item() * norm

    推荐环境 torch1.3 python3.6
"""
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
        mask: 掩码张量 ? [2, 4, 4]
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
# print(f'attn:{attn}\n{attn.shape}\np_attn:{p_attn}\n{p_attn.shape}')

# x = torch.randn(4, 4)
# print(x.size(), x.shape)
# y = x.view(16)
# print(y.size())
# z = x.view(-1, 8)
# print(z.shape)

# a = torch.randn(1, 2, 3, 4)
# print(a.shape)
# print(a)
# b = a.transpose(1, 2)
# print(b.shape)
# print(b)
#
# c = a.view(1, 3, 2, 4)      # transpose()、view()形状虽然相同，但结果并不同
# print(c.shape)
# print(c)


# 实现深层拷贝/克隆函数，因为多头注意力机制下，要用到多个结构相同的线性层
# 需要使用clone()函数将他们一同初始化到一个网络层列表对象中
def clones(module, N):
    """
    module: 代表要克隆的目标网络层
    N: 将module克隆几个
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力机制类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=.1):
        """
        :param head: 代表几个头
        :param embedding_dim: 词嵌入维度
        :param dropout: 置零比率
        """
        super(MultiHeadedAttention, self).__init__()

        # 要确认一个事实：多头的数量head需要整除词嵌入维度embedding_dim
        assert embedding_dim % head == 0

        # 得到每个头获得的词向量维度
        self.d_k = embedding_dim // head

        self.head = head
        self.embedding_dim = embedding_dim

        # 获得线性层，要获得4个，分别是Q K V以及最终的输出线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化注意力张量
        self.attn = None

        # 初始化dropout对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ query key value是多头注意力机制的三个输入张量
            mask 掩码张量 """
        # 首先判断是否使用掩码张量
        if mask is not None:
            # 使用squeeze将掩码张量进行维度扩充，代表多头中的第n个头
            # 【这里我怀疑教程讲错了，教程升维是[1]，运行报错；我改为[0]，可正常运行】
            mask = mask.unsqueeze(0)

        # 获得batch_size
        batch_size = query.size(0)

        # 首先使用zip将网络层和输入数据连接在一起，模型的输出利用view和transpose进行维度和形状的改变
        # for循环中model(x)的结果要view()改变形状[batch_size, tokens词的数量/句子长度, 多少个头, 每个头分得的维度]
        # transpose(), 因为想把头的数量往前，-1与词嵌入维度连接到一起
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                for model, x in zip(self.linears, (query, key, value))]

        # 将每个头的输出传入到注意力层
        # for循环将输入的qkv经过fc，均分到每个head，然后送入以下attention()，得到多头注意力计算结果
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 得到每个头的计算结果是4维张量，需要进行形状的转换
        # 前面已经将1 2两个维度进行过转置，这里要重新转置回来
        # 注意：经历了transpose()后，必须要用contiguous()，不然无法使用view()
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后将x输入最后一个线性层，得到最终的多头注意力结构输出
        return self.linears[-1](x)


# 实例化若干参数
head = 8    # 8个头均分512维处理
embedding_dim = 512
dropout = .2

# 输入参数
query = key = value = pe_result     # [2, 4, 512]

mask = Variable(torch.zeros(8, 4, 4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
# print(mha_result, mha_result.shape)

# 构建前馈全连接网络类
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        """
        :param d_model: 词嵌入维度，同时也是两个线性层的输入维度和输出维度
        :param d_ff: 第一个线性层的输出维度，和第二个线性层的输入维度
        :param dropout: 随机置零比率
        """
        super(PositionWiseFeedForward, self).__init__()

        # 定义两层全连接的fc
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ x: 来自上一层的输出 """
        # 首先将x送入第一个fc，然后经历relu，再经历dropout
        # 最后送入第二个fc
        return self.w2(self.dropout(F.relu(self.w1(x))))


d_model = 512
d_ff = 64
dropout = .2

x = mha_result
ff = PositionWiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
# print(ff_result)
# print(ff_result.shape)

# 构建规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features: 词嵌入维度
        :param eps: 一个足够小的正数
        """
        super(LayerNorm, self).__init__()

        # 初始化两个参数张量 a2、b2，用于对结果做规范化计算
        # 将其用nn.Parameter封装，代表也是模型参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        """ x: 上一层网络的输出 """
        # 首先对x求最后维度上的均值，同时keepdim
        mean = x.mean(dim=-1, keepdim=True)     # [2, 4, 512] -> [2, 4, 1] 如果没有keepdim=True，得到的形状是[2, 4]
        # 接着对x求最后维度上的标准差，同时keepdim
        std = x.std(dim=-1, keepdim=True)       # [2, 4, 512] -> [2, 4, 1]
        # 按照规范化公式进行计算并返回
        # x形状[2, 4, 512] mean形状[2, 4, 1], 也就是x的512维度每个数都减去mean的为1维度, 广播机制。确实是对batch的每个样本操作，符合LN
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model = 512
eps = 1e-6

x = ff_result   # [2, 4, 512]
ln = LayerNorm(features, eps)
ln_result = ln(x)
# print(ln_result, ln_result.shape)

# 构建子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=.1):
        """
        :param size: 词嵌入维度
        :param dropout: 置零比率
        """
        super(SublayerConnection, self).__init__()
        # 实例化一个规范化层的对象
        self.norm = LayerNorm(size)
        # 实例化一个dropout对象
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, sublayer):
        """
        :param x: 上一层传入的张量
        :param sublayer: 该子层连接中的子层函数
        :return:
        """
        # 首先将x进行规范化，然后送入子层函数处理，处理结果进入dropout层，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))

size = d_model = 512
head = 8
dropout = .2

# x = pe_result       # [2, 4, 512]
# mask = Variable(torch.zeros(8, 4, 4))
# self_attn = MultiHeadedAttention(head, d_model)
#
# sublayer = lambda x: self_attn(x, x, x, mask)
#
# sc = SublayerConnection(size, dropout)
# sc_result = sc(x, sublayer)     # [2, 4, 512]
# print(sc_result, sc_result.shape)

# 构建编码器层的类
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度
        :param self_attn: 多头自注意力子层的实例化对象
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout: 置零比率
        """
        super(EncoderLayer, self).__init__()

        # 将两个实例化对象和参数传入类中
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        # 编码器层中有2个子层连接结构，使用clone()
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        :param x: 上一层的传入张量
        :param mask: 掩码张量
        :return:
        """
        # 首先让x经过第一个子层连接结构，内部包含多头自注意力机制子层
        # 再让张量经过第二个子层连接结构，其中包含前馈全连接网络
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# size = d_model = 512
# head = 8
# d_ff = 64
# x = pe_result
# dropout = .2
#
# self_attn = MultiHeadedAttention(head, d_model)
# ff = PositionWiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(torch.zeros(8, 4, 4))
#
# el = EncoderLayer(size, self_attn, ff, dropout)
# el_result = el(x, mask)     # [2, 4, 512]
# print(el_result, el_result.shape)

# 构建编码器类Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 编码器层
        :param N: 编码器中有几个layer
        """
        super(Encoder, self).__init__()

        # 首先使用clones()克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 初始化一个规范化层，作用在编码器的最后面
        self.norm = LayerNorm(layer.size)       # layer.size就是词嵌入维度

    def forward(self, x, mask):
        # 让x依次经历N个编码器层，最后经过规范化层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


size = d_model = 512
d_ff = 64
head = 8
c = copy.deepcopy
attn =  MultiHeadedAttention(head, d_model)
ff = PositionWiseFeedForward(d_model, d_ff, dropout)
dropout = .2
layer = EncoderLayer(size, c(attn), c(ff), dropout)
N = 8
mask = Variable(torch.zeros(8, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)     # [2, 4, 512]
# print(en_result, en_result.shape)


# 构建解码器层的类
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度
        :param self_attn: 多头自注意力机制实例化对象
        :param src_attn: 常规注意力机制实例化对象
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout: 置零比率
        """
        super(DecoderLayer, self).__init__()

        # 将参数传入类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        # 按照解码器层的结构图，使用clone()克隆3个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层输入的张量
        :param memory: 编码器的语义存储张量
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return:
        """
        m = memory

        # 第一步让x经历第一个子层，多头自注意力机制的子层
        # 采用target_mask，为了将解码时未来的信息进行遮掩，比如模型解码第2个字符，只能看见第一个字符信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 第二步让x经历第二个子层，常规注意力机制子层，Q!=K=V
        # 采用source_mask，为了遮掩掉对结果信息无用的数据
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 第三步让x经历第三个子层，前馈全连接层
        return self.sublayer[2](x, self.feed_forward)

size = d_model = 512
head = 8
d_ff = 64
dropout = .2

self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)

ff = PositionWiseFeedForward(d_model, d_ff, dropout)

x = pe_result       # [2, 4, 512]

memory = en_result      # [2, 4, 512]

mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_reslut = dl(x, memory, source_mask, target_mask)     # [2, 4, 512]
# print(dl_reslut, dl_reslut.shape)

# 构建解码器类
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 解码器层实例化对象
        :param N:
        """
        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        # 实例化一个规范化层
        self.norm = LayerNorm(layer.size)       # layer.size大概率词嵌入维度

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 目标数据的嵌入表示
        :param memory: 编码器的输出张量
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        """
        # 要将x依次经历所有的编码器层处理，最后通过规范化层
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

size = d_model = 512
head = 8
d_ff = 64
dropout = .2
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionWiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)

N = 8
x = pe_result       # 位置嵌入结果
memory = en_result  # 编码结果
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
# print(de_result, de_result.shape)

# 构建Generator类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 代表词嵌入的维度
        :param vocab_size: 代表词表大小
        """
        super(Generator, self).__init__()
        # 定义一个fc，完成网络输出维度变换
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """ x: 上一层输出张量 """
        # 首先将x送入fc，让其经历softmax处理
        return F.log_softmax(self.project(x), dim=-1)


d_model = 512
vocab_size = 1000
x = de_result

gen = Generator(d_model, vocab_size)
gen_result = gen(x)     # [2, 4, 1000]  1000 == vocab_size，要进行最后的token分类了
# print(gen_result, gen_result.shape)

# 构建 编码器-解码器 类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        :param encoder: 编码器实例化对象
        :param decoder: 解码器实例化对象
        :param source_embed: 源数据的嵌入函数
        :param target_embed: 目标数据的嵌入函数
        :param generator: 输出部分的类别生成器实例化对象
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        :param source: 源数据  并未经历词嵌入
        :param target: 目标数据 并未经历词嵌入
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据的掩码张量
        :return:
        """
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        """ 完成 原始输入的 嵌入+编码 """
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """
        解码输入：目标嵌入+编码最终输出，掩码
        :param memory: 经历编码器后的输出张量
        """
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
d_model = 512
encoder = en    # 编码器
decoder = de    # 解码器
source_embed = nn.Embedding(vocab_size, d_model)    # 输入数据嵌入函数
target_embed = nn.Embedding(vocab_size, d_model)    # 输出数据嵌入函数
generator = gen

source = target = Variable(torch.LongTensor([[100, 2, 421, 508],
                                             [491, 998, 1, 221]]))
source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)        # [2, 4, 512]
# print(ed_result, ed_result.shape)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=.1):
    """
    构造完整的Transformer模型
    :param source_vocab: 源数据的词汇总数
    :param target_vocab: 目标数据的词汇总数
    :param N: 编码器、解码器堆叠的层数
    :param d_model: 词嵌入维度
    :param d_ff: 前馈全连接层中变化矩阵维度
    :param head: 多头注意力机制的头数
    :param dropout: 置零比率
    """
    c = copy.deepcopy

    # 实例化多头注意力的类
    attn = MultiHeadedAttention(head, d_model)

    # 实例化一个前馈全连接层网络
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)

    # 实例化一个位置编码器
    position = PositionalEncoding(d_model, dropout)

    # 实例化模型model，利用的是EncoderDecoder类
    # 编码器结构有2个子层——attention层和前馈全连接层
    # 解码器结构有3个子层——两个attention层和一个前馈全连接层
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    # 初始化整个模型参数，判断参数维度>1则将矩阵初始化服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


source_vocab = 11
target_vocab = 11
N = 6
#
# if __name__ == "__main__":
#     res = make_model(source_vocab, target_vocab, N)
#     print(res)


from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute, run_epoch, greedy_decode

def data_generator(V, batch_size, num_batch):
    """
    :param V: 随机生成数据的最大值+1
    :param batch_size: 样本数量
    :param num_batch: 一共有多少个batch_size的数据
    :return:
    """
    for i in range(num_batch):
        # 使用random.randint()随机生成[1, V)
        # 分布的形状(batch, 10)
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10), dtype='int64'))

        # 将数据第一列置1，作为起始标志
        data[:, 0] = 1

        # 因为是copy任务，所以源数据和目标数据完全一致
        # 设置参数retuires_grad=False, 样本的参数不需要参与梯度计算
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)


V = 11
batch_size = 20
num_batch = 30

# 使用make_model()获得模型实例化对象
model = make_model(V, V, N=2)

# 使用工具包get_std_opt获得模型优化器
model_optimizer = get_std_opt(model)

# 使用工具包LabelSmoothing获得标签平滑对象
# size 目标词汇总数，这里默认与源数据词汇总数相等；smoothing表示标签的平滑程度，如原来标签的表示值为1，则平滑后他的值域变为[1-smoothing, 1+smoothing]
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)    # criterion 标准、尺度、规则、规范

# 使用工具包SimpleLossCompute获得利用标签平滑的结果得到的损失计算
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


crit = LabelSmoothing(size=5, padding_idx=0, smoothing=.5)
predict = Variable(torch.FloatTensor([[0, .2, .7, .1, 0],
                                      [0, .2, .7, .1, 0],
                                      [0, .2, .7, .1, 0]]))
target = Variable(torch.LongTensor([2, 1, 0]))
# 黄色方块，它相对于横坐标横跨的值域就是标签平滑后的正向平滑值域，我们可以看到大致是.5到2.5
#         它相对于纵坐标横跨的值域就是标签平滑后的负向平滑值域，我们可以看到大致是-.5到1.5，总的值域空间由原来的[0,2]变为[-.5, 2.5]
crit(predict, target)
# plt.imshow(crit.true_dist)
# plt.show()

# def run(model, loss, epochs=10):
#     """
#     :param model: 训练的模型
#     :param loss: 使用的损失计算方法
#     :param epochs:
#     """
#     for epoch in range(epochs):
#         # 首先进入训练模式，所有的参数都会被更新
#         model.train()
#         # 训练时，传入的batch_size是20
#         run_epoch(data_generator(V, 8, 20), model, loss)
#
#         # 训练结束后，进入评估模式，所有参数固定不变
#         model.eval()
#         # 评估时，传入的batch_size是5
#         run_epoch(data_generator(V, 8, 5), model, loss)
#
#
# if __name__ == '__main__':
#     run(model, loss)

def run(model, loss, epochs=10):
    for epoch in range(epochs):
        # 首先进入训练模式 所有的参数将会被更新
        model.train()

        run_epoch(data_generator(V, 8, 20), model, loss)

        # 训练结束后，进入评估模式，所有参数不更新
        model.eval()

        run_epoch(data_generator(V, 8, 5), model, loss)

    # for结束代表训练结束，再次进入评估模式
    model.eval()

    # 初始化一个输入张量
    source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]))

    # 初始化一个输入张量的掩码张量，全1代表没有任何遮掩
    source_mask = Variable(torch.ones(1, 1, 10))    # 前两个1表示扩充维度

    # 设定解码的最大长度max_len=10，起始数字标志默认等于1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


# if __name__ == '__main__':
#     run(model, loss)

