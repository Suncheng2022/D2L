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

""" 2.2输入部分的实现 """
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
        return self.lut(x) * math.sqrt(self.d_model)  # 要进行一下缩放


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
        x += Variable(self.pe[:, :x.size(1), :], requires_grad=False)  # 位置编码是不参与训练
        return self.dropout(x)


""" 2.3 编码器部分的实现 """
def subsequent_mask(size):
    """
    构建掩码张量函数 返回tensor下三角阵 左下方有值; 1表示遮掩 看不见，0不遮掩 看得见，具体含义由程序员定义
    :param size: 代表掩码张量后两个维度 形成一个方阵
    :return: 形状 [1, size, size]
    """
    attn_shape = (1, size, size)
    # 先构建全1张量，使用np.triu()转为上三角阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 三角阵 反转为下三角阵
    return torch.from_numpy(1 - subsequent_mask)


def attention(query, key, value, mask=None, dropout=None):
    """
    query key value：注意力的三个输入张量; 传参时三者传的是同一个变量 [句子数，词数，嵌入维度d_model]
    mask：掩码张量
    dropout: 不是置零比率，而是dropout层实例化对象
    """
    # 取query最后一个维度 词嵌入维度
    d_k = query.size(-1)
    # 按注意力公式计算，将query和key的转置进行矩阵乘法，然后除以缩放系数
    # transpose(dim1, dim2)会直接交换2个维度，也只能操作2个维度，无论这2个维度参数的顺序
    # print(f'$$$$$$$$$$$$$$$$$ query {query.shape} key {key.shape} val {value.shape}')   # [2, 4, 512]
    # [2, 4, 512] --> key.transpose(-2, -1) --> [2, 512, 4]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    # [2, 4, 4]
    # 判断是否使用掩码张量mask
    if mask is not None:
        # 利用masked_fill()，将掩码张量mask和0进行位置的一一比较，如果等于0则替换成非常小的数
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对scores最后维度进行softmax操作
    # print(f'$$$$$$$$$$$$$$$$$ scores {scores.shape}')     # [2, 4, 4]
    p_attn = F.softmax(scores, dim=-1)  # [2, 4, 4]
    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后一步，完成p_attn与value的乘法，并返回query的注意力表示p_attn
    return torch.matmul(p_attn, value), p_attn      # [2, 4, 4] * [2, 4, 512] = [2, 4, 512]; [2, 4, 4]


def clones(module, N):
    """
    使用copy.deepcopy()深拷贝出N个相同的module
    因为多头注意力需要使用多个结构相同但参数不同的线性层
    :param module: 要深拷贝的模块
    :param N: 深拷贝的数量
    :return: nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    """ 实现 多头注意力机制 类 """
    def __init__(self, num_head, embedding_dim, dropout=.1):
        """
        猜测：一个MultiHeadAttention类对象应该只实现了一个注意力头，可能要实例化多个MultiHeadAttention类对象
        :param num_head: 注意力头的数量
        :param embedding_dim: 词嵌入维度
        :param dropout: 置零比率
        """
        super(MultiHeadAttention, self).__init__()
        # 要确认一个事实，embedding_dim能被num_head整除，即每个头要拿到相同数量的词向量维度进行处理
        assert embedding_dim % num_head == 0
        # 每个head获得的词向量维度
        self.d_k = embedding_dim // num_head

        self.num_head = num_head
        self.embedding_dim = embedding_dim

        # 初始化线性层，给Q K V以及最终的输出使用，共4个
        self.linears = clones(nn.Linear(self.embedding_dim, self.embedding_dim), 4)
        # 初始化注意力张量
        self.attn = None
        # 初始化dropout层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        query key val是注意力机制的3个输入张量，mask表示掩码张量
        """
        # 首先判断是否使用mask
        if mask is not None:
            # 对mask掩码张量扩充维度，表示多头中的第n个头
            # print(f'$$$$$$$$$ mask {mask.shape}')
            mask = mask.unsqueeze(1)
        # 获取batch_size，即样本个数
        batch_size = query.size(0)      # 等价于.size()[0], .shape[0]
        # 使用zip将网络层和输入数据连接在一起，不同的网络层处理不同的数据嘛；使用view()和transpose()对输出维度和形状进行改变
        # model(x)的维度进行改变 [embedding_dim]->[batch_size, -1(词汇长度), self.num_head, self.d_k]
        # transpose交换维度，我们希望num_head往前，词汇长度 与 每个头分得的嵌入维度 挨着 [batch_size, -1, self.num_head, self.d_k]->[batch_size, self.num_head, -1, self.d_k]
        query, key, value = \
            [model(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]   # 这里使用了3个fc，还有1个fc后面使用
        # 每个头得到的qkv送入注意力层
        x, self.attn = attention(query, key, value, mask, self.dropout)
        # 得到每个头的结果是4维张量，需要进行结果的转换
        # 前面已经将1,2两个维度进行过转换，这里再转换回来
        # transpose()后必须使用contiguous()方法，才能使用view()
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        # 进入最后一个fc
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """ 构建 前馈全连接网络 类 """
    def __init__(self, d_model, d_ff, dropout=.1):
        """
        :param d_model: 词嵌入维度
        :param d_ff: 第一个fc的输出维度、第二个fc的输入维度
        :param dropout: 随机置零比率
        """
        super(PositionwiseFeedForward, self).__init__()
        # 定义2个fc层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """ x：上一层即多头注意力层的输出 """
        # 首先经过fc1、relu、dropout，再经过fc2
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    """ 构建规范化层的类 """
    def __init__(self, d_model, eps=1e-6):
        """
        :param d_model: 词嵌入的维度
        :param eps: 很小的正数，防止除零
        """
        super(LayerNorm, self).__init__()
        # 初始化两个参数张量a2、b2，用于对结果做规范化操作计算
        # 使用nn.Parameter()封装，参加训练更新
        self.a2 = nn.Parameter(torch.ones(d_model))     # 这两个参数应该是规范化层的gamma、beta
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x：上一层网络的输出
        # 首先对x最后一个维度求均值，keepdim=true保持输入输出维度一致(求均值的维度变成1了，而不是消失)
        mean = x.mean(dim=-1, keepdim=True)
        # 然后对x最后一个维度求标准差，keepdim=true保持输入输出维度一致
        std = x.std(dim=-1, keepdim=True)
        # 按LN规范化公式计算并返回
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    """ 构建子层连接结构的类 """
    def __init__(self, d_model, dropout=.1):
        """
        :param d_model: 词嵌入维度
        :param dropout: 随机置零比率
        """
        super(SublayerConnection, self).__init__()
        # 实例化规范化层
        self.norm = LayerNorm(d_model)
        # 实例化dropout层
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, sublayer):
        """
        x：上一层的输出张量
        sublayer: 该子层连接中子层函数
        """
        # 规范化、送入子层、dropout、残差连接
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """ 构建编码器层的类 """
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        """
        :param d_model: 词嵌入维度
        :param self_attn: 多头自注意力层实例化对象
        :param feed_forward: 前馈网络层实例化对象
        :param dropout: 随机置零比率
        """
        super(EncoderLayer, self).__init__()
        # 将两个实例化对象和参数传入类中
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.d_model = d_model

        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        """
        :param x: 代表上一层的输出
        :param mask: 代表掩码张量
        """
        # 首先经过第一个子层连接结构，内部包含多头自注意力机制子层
        # 然后经过第二个子层连接结构，内部包含前馈全连接层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """ 构建编码器类 """
    def __init__(self, layer, N):
        """
        :param layer: 代表编码器层
        :param N: 编码器中有几个layer
        """
        super(Encoder, self).__init__()
        # 使用clones()函数克隆N个编码器放在self.layers中
        self.layers = clones(layer, N)
        # 初始化规范化层，作用在编码器最后面
        # print(f'$$$$$$$$$$$$$$$$ 可能报错 layer.d_model {layer.d_model}')
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """
        :param x: 上一层的输出
        :param mask: 掩码张量
        """
        # x依次经过N个编码器的处理，最后经过规范化层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """ 构建解码器层的类 """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度
        :param self_attn: 多头自注意力机制的实例化对象
        :param src_attn: 常规的注意力机制的实例化对象
        :param feed_forward: 前馈全连接层的实例化对象
        :param dropout: 随机置零比率
        """
        super(DecoderLayer, self).__init__()
        # 将参数传入
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 按照解码器层的结构，克隆3个子层连接对象
        self.sublayer = clones(SublayerConnection(self.size, self.dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层输入张量
        :param memory: 编码器的语义存储张量
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        """
        m = memory
        # 第一步让x经历第一个子层，多头自注意力的子层
        # 采用target_mask，解码时遮掩未来的信息，比如模型解码第二个字符时，只能看见第一个字符信息，后面的不能看到
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 第二步让x经历第二个子层，常规注意力层，Q!=K==V
        # 采用source_mask，为了遮掩掉对结果信息无用的数据
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        # 第三步让x经历第三个子层，前馈全连接层
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """ 构建解码器类 """
    def __init__(self, layer, N):
        """
        :param layer: 代表解码器层的实例化对象
        :param N: 代表将layer进行N次拷贝
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 代表目标数据的嵌入表示
        :param memory: 代表编码器的输出张量
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        """
        # x依次经过所有编码器层的处理，最后经过规范化层
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


""" 2.5输出部分的实现 """
class Generator(nn.Module):
    """ 构建Generator类 """
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词嵌入维度
        :param vocab_size: 词表大小
        """
        super(Generator, self).__init__()
        # 定义线性层，作用是完成网络输出维度的转换
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """ x：上一层输出的张量 """
        return F.log_softmax(self.project(x), dim=-1)


""" 2.6模型的构建 """

if __name__ == '__main__':
    """ 2.3 编码器部分的实现 """
    # 测试np.triu()
    # x = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], k=-1)  # k=0表示对角线 保留 不置零，k=-1表示下移一位
    # y = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], k=0)
    # z = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], k=1)
    # print(x, y, z, sep='\n')

    # 生成掩码
    # size = 5
    # sm = subsequent_mask(size)
    # print(sm, sm.shape)     # torch.Size([1, 5, 5]
    # plt.figure(figsize=(5, 5))
    # plt.imshow(subsequent_mask(20).squeeze(0))
    # plt.show()

    # x = Variable(torch.randn(5, 5))
    # print(x)
    # mask = Variable(torch.zeros(5, 5))
    # print(mask)
    # y = x.masked_fill(mask == 0, -1e9)
    # print(y)

    # 测试attention()
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

    # query = key = value = x_emb_pe
    # mask = Variable(torch.zeros(2, 4, 4))
    # attn, p_attn = attention(query, key, value, mask=mask)
    # print(f'attn {attn} {attn.shape}\np_attn {p_attn} {p_attn.shape}')

    # view() transpose()都能交换维度，shape同，但结果不同
    # x = torch.randn(4, 4)
    # y = x.view(16)
    # z = x.view(-1, 8)   # -1表示维度自动推断
    # # print(x.shape, y.shape, z.shape)
    #
    # a = torch.randn(1, 2, 3, 4)
    # b = a.transpose(1, 2)   # 交换a的索引1、2维度，即形状[1, 2, 3, 4]->[1, 3, 2, 4]
    # c = a.view(1, 3, 2, 4)  # 使用view()也能达到b的目的; 但是——transpose()和view()都能达到交换维度的作用，但方式不同！！！
    # print(a, a.shape, b, b.shape, c, c.shape, sep='\n')

    # 测试多头注意力
    # 实例化若干参数
    num_head = 8
    embedding_dim = 512
    dropout = .2
    # 若干输入参数的初始化
    query = key = value = torch.repeat_interleave(x_emb_pe, repeats=4, dim=0)
    # print(f'$$$$$$$$ query {query.shape}')      # [8, 4, 512]
    mask = Variable(torch.zeros(8, 4, 4))       # 维度要和qkv等相同

    mha = MultiHeadAttention(num_head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)
    # print(mha_result, mha_result.shape, sep='\n')

    # 测试前馈全连接网络类
    d_model = 512
    d_ff = 64
    dropout = .2
    x = mha_result
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_result = ff(x)
    # print(ff_result, ff_result.shape, sep='\n')

    # 测试规范化层类
    d_model = 512
    eps = 1e-6
    x = ff_result
    ln = LayerNorm(d_model, eps)
    ln_result = ln(x)
    # print(ln_result, ln_result.shape, sep='\n')

    # 测试子层连接结构的类
    d_model = 512
    head = 8
    dropout = .2

    x = torch.repeat_interleave(x_emb_pe, repeats=4, dim=0)
    mask = Variable(torch.zeros(8, 4, 4))
    self_attn = MultiHeadAttention(head, d_model)

    sublayer = lambda x: self_attn(x, x, x, mask)

    sc = SublayerConnection(d_model, dropout)
    sc_result = sc(x, sublayer)
    # print(sc_result, sc_result.shape, sep='\n')

    # 测试编码层类
    d_model = 512
    head = 8
    d_ff = 64
    x = torch.repeat_interleave(x_emb_pe, repeats=4, dim=0)
    dropout = .2

    self_attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))

    el = EncoderLayer(d_model, self_attn, ff, dropout)
    el_result = el(x, mask)
    # print(el_result, el_result.shape)

    # 测试编码器类
    d_model = 512
    d_ff = 64
    head = 8
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    dropout = .2
    layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    N = 8
    mask = Variable(torch.zeros(8, 4, 4))

    en = Encoder(layer, N)
    en_result = en(x, mask)
    # print(f'编码器类输出 {en_result}\n{en_result.shape}')

    # 测试解码器层类
    size = d_model = 512
    head = 8
    d_ff = 64
    dropout = .2

    self_attn = src_attn = MultiHeadAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    x = pe_result = torch.repeat_interleave(x_emb_pe, repeats=4, dim=0)
    memory = en_result
    mask = Variable(torch.zeros(8, 4, 4))
    source_mask = target_mask = mask

    dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
    dl_reslut = dl(x, memory, source_mask, target_mask)
    # print(dl_reslut, dl_reslut.shape, sep='\n')

    # 测试解码器层的类
    size = d_model = 512
    head = 8
    d_ff = 64
    dropout = .2
    c = copy.deepcopy
    attn = MultiHeadAttention(head, size)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(size, c(attn), c(attn), c(ff), dropout)

    N = 8
    x = pe_result
    memory = en_result
    mask = Variable(torch.zeros(8, 4, 4))
    source_mask = target_mask = mask

    de = Decoder(layer, N)
    de_result = de(x, memory, source_mask, target_mask)
    # print(de_result, de_result.shape)

    """ 2.5输出部分的实现 """
    d_model = 512
    vocab_size = 1000
    x = de_result

    gen = Generator(d_model, vocab_size)
    gen_result = gen(x)
    print('gen_result', gen_result, gen_result.shape)

    """ 2.6模型的构建 """


    """ 2.2输入部分的实现 """
    # d_model = 512
    # vocab = 1000
    # x = Variable(torch.LongTensor([[100, 2, 421, 508],
    #                                [491, 998, 1, 221]]))
    # emb = Embeddings(d_model, vocab)
    # output = emb(x)
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
    # d_model = 512
    # vocab = 1000
    # dropout = .1
    # max_len = 60
    #
    # emb_layer = Embeddings(d_model, vocab)
    # pe_layer = PositionEmbedding(d_model, dropout, max_len)
    # x = Variable(torch.LongTensor([[100, 2, 421, 508],
    #                                [491, 998, 1, 221]]))
    # x_emb = emb_layer(x)
    # x_emb_pe = pe_layer(x_emb)
    # print(x_emb_pe, x_emb_pe.shape)     # torch.Size([2, 4, 512]
    #
    # # 可视化位置编码信息
    # # 设置画布
    # plt.figure(figsize=(15, 5))
    # # 实例化PositionEmbedding类，dropout设为0是为了把位置编码全显示出来
    # pe_layer = PositionEmbedding(d_model=20, dropout=0.)
    # # 向pe中传入全零向量，为的是只显示编码位置信息
    # y = pe_layer(Variable(torch.zeros(1, 100, 20)))     # 本应传入词嵌入结果
    # # 画数据
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    # plt.show()      # 奇数开头的从同一起点开始，意为使用的同一sin/cos编码；偶数开头的同
