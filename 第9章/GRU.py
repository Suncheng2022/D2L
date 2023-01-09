"""
2023.01.02
尝试手动实现NLP相关模型--没搞懂...
"""
import torch
import argparse
import random

from torch import nn
from torch.nn import functional as F
from collections import Counter
from tqdm import tqdm


class RNNModel:
    """ 暂时不继承nn.model，先跟着教程实现 """
    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 初始化RNN单元的参数
        num_inputs = num_outputs = self.vocab_size
        W_xz, W_hz, b_z = torch.randn((num_inputs, self.num_hiddens), device=device) * .01, \
                            torch.randn((self.num_hiddens, self.num_hiddens), device=device) * .01, \
                            torch.zeros(self.num_hiddens, device=device)    # 更新门参数
        W_xr, W_hr, b_r = torch.randn((num_inputs, self.num_hiddens), device=device) * .01, \
                            torch.randn((self.num_hiddens, self.num_hiddens), device=device) * .01,\
                            torch.zeros(self.num_hiddens, device=device)    # 重置门参数
        W_xh, W_hh, b_h = torch.randn((num_inputs, self.num_hiddens), device=device) * .01, \
                            torch.randn((self.num_hiddens, self.num_hiddens), device=device) * .01, \
                            torch.zeros(self.num_hiddens, device=device)    # 候选隐状态参数
        W_hq = torch.randn((self.num_hiddens, num_outputs), device=device)
        b_q = torch.zeros(num_outputs, device=device)
        self.params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in self.params:
            param.requires_grad_(True)
        
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = self.params
        H = state
        outputs = []
        for x in X:
            Z = torch.sigmoid((x @ W_xz) + (H @ W_hz) + b_z)
            R = torch.sigmoid((x @ W_xr) + (H @ W_hr) + b_r)
            H_tilda = torch.tanh((x @ W_xh) + ((R * H) @ W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), H


class Vocab:
    """ 
    功能：处理text的vocabulary 
    参数：tokens 一行一行按word/char拆分的文本行
    """
    def __init__(self, tokens=None, min_freq=0):
        """
        1.将tokens处理为 {token:frequency, ...}，按frequency降序排序
        2.创建id与token的互相映射
        """
        self._token_frequency = Counter([token for line in tokens for token in line])    # python内置函数，返回key:frequency
        self._token_frequency = dict(sorted(self._token_frequency.items(), key=lambda x: x[1], reverse=True))        

        # 初始化id与token的互相映射，后面会继续添加
        self.id_to_token = ['<unk>']    # 索引即id
        self.token_to_id = {token: id for id, token in enumerate(self.id_to_token)}
        for token, freq in self._token_frequency.items():
            if freq < min_freq:
                continue    # 这里填break也可以，因为self.token_frequency已经按freq从大到小排序了
            if token not in self.id_to_token:
                self.id_to_token.append(token)      # 注意self.id_to_token与self.token_to_id的关系
                self.token_to_id[token] = len(self.id_to_token) - 1

    def __len__(self):
        """ 有多少个token(word or char) """
        return len(self.id_to_token)
    
    def __getitem__(self, tokens):
        """ 返回token(s)的 索引
            Vocab类，从名字上的意义就是处理word or char，getitem是拿词对应的索引 """
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_id[tokens]
        return [self.token_to_id[token] for token in tokens]
    
    def to_token(self, inds):
        if not isinstance(inds, (tuple, list)):
            return self.id_to_token[inds]
        return [self.id_to_token[ind] for ind in inds]
    
    @property
    def unk(self):
        """ @property修饰，使unk()方法的使用变成同访问属性一样 """
        return 0
    
    @property
    def token_frequency(self):
        """ @property修饰，使方法调用同访问属性一样 """
        return self.token_frequency


def get_data_batch(curpus, batch_size, num_steps):
    """ 由语料库curpus取出num_tokens个token，然后每调用一次方法返回一个batch个数据 """
    offset = random.randint(0, num_steps)
    num_tokens = ((len(curpus) - offset) // batch_size) * batch_size
    Xs = torch.tensor(curpus[offset: offset + num_tokens]).reshape(batch_size, -1)
    Ys = torch.tensor(curpus[offset + 1: offset + num_tokens + 1]).reshape(batch_size, -1)
    num_batches = Xs.shape[-1] // batch_size    # 计算有多少个batch，因为Xs.shape=[batch_size, -1]，-1的这个维度是多少个batch，理解没
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


def tokenize(lines, token='word'):
    """ 将lines的每一行按word/char拆分 
        返回：还是一行一行的，只不过按需拆分了 """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print(f'Error:Not such token type:{token}')


def read_time_machine():
    """ 功能：txt文本过滤掉特殊字符、转为小写 """
    with open('../data/timemachine.txt', 'r') as file:
        lines = file.readlines()

        lines_after_filt = []
        for line in lines:
            tmp = ''
            for c in line:
                if 97 <= ord(c) <= 97 + 25 or 65 <= ord(c) <= 65 + 25 or c == ' ':
                    tmp += c
            lines_after_filt.append(tmp.lower())
                    
    return lines_after_filt


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def main():
    parser = argparse.ArgumentParser(description='GRU by sc. 2023.01.05')
    parser.add_argument('--max_tokens', default=10000, type=int, help='')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--num_steps', default=35, type=int, help='')
    parser.add_argument('--num_epochs', default=500, type=int, help='')
    parser.add_argument('--num_hiddens', default=256, type=int, help='')
    parser.add_argument('--device', default='0', help='')
    parser.add_argument('--lr', default=1., type=float, help='')
    opt = parser.parse_args()
    device = torch.device(f'cuda:{opt.device}') if torch.cuda.device_count() else torch.device(f'cpu')
    print(f'training on {device}')

    # 数据预处理
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    curpus = [vocab[token] for line in tokens for token in line]    # 通过Vocab类将token转为数字索引
    curpus = curpus[:opt.max_tokens]

    # Dataloader--sequence data
    train_iter = get_data_batch(curpus, opt.batch_size, opt.num_steps)
    
    # train
    vocab_size = len(vocab)
    net = RNNModel(vocab_size, opt.num_hiddens, device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.params, opt.lr)
    # optimizer = lambda batch_size: sgd(net.params, opt.lr, 1)
    
    for epoch in tqdm(range(opt.num_epochs)):
        state = torch.zeros((opt.batch_size, opt.num_hiddens), device=device)
        for X, Y in train_iter:
            optimizer.zero_grad()
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state)
            l = loss(y_hat, y.long()).mean()
            l.backward()
            # optimizer(1)
            optimizer.step()


if __name__ == '__main__':
    main()
