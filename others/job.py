a = 1
b = 2
print(a + b)





# from turtle import forward
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Attn(nn.Module):
#     def __init__(self, qurey_size, key_size, value_size1, value_size2, output_size) -> None:
#         super(Attn, self).__init__()
#         self.query_size = qurey_size
#         self.key_size = key_size
#         self.value_size1 = value_size1
#         self.value_size2 = value_size2
#         self.output_size = output_size

#         # 初始化注意力机制实现中第一步的线性层
#         self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)
#         # 初始化注意力机制实现中第三步线性层
#         self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)

#     def forward(self, Q, K, V):
#         # 注意，假定Q,K,V都是三维张量
#         # 第一步，将Q,K进行纵轴上的拼接-->线性变换-->softmax处理得到注意力向量
#         attn_weights = F.softmax(self.attn(torch.cat((Q[0], K[0]), dim=1)), dim=1)
#         # 第二步，将注意力矩阵和V进行bmm运算
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)
#         # 取Q[0]进行降维，再次和运算结果进行拼接
#         output = torch.cat((Q[0], attn_applied[0]), dim=1)
#         # 第三步，将上面的输出进行一次线性变换然后扩展维度成3维张量
#         output = self.attn_combine(output).unsqueeze(0)
#         return output, attn_weights


# qurey_size = 32
# key_size = 32
# value_size1 = 32
# value_size2 = 64
# output_size = 64

# attn = Attn(qurey_size, key_size, value_size1, value_size2, output_size)
# Q = torch.randn(1, 1, 32)
# K = torch.randn(1, 1, 32)
# V = torch.randn(1, 32, 64)
# output = attn(Q, K, V)
# print(output[0], output[0].size())
# print(output[1], output[1].size())