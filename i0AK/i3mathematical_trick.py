import torch

# https://www.bilibili.com/video/BV1v4421c7fr/?vd_source=4ead76717cce1a9fe9f2f10530a28c24

# 设定随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 创建一个 3x3 的下三角矩阵，元素全为1
a = torch.tril(torch.ones(3, 3))

# 将每个元素除以其所在行的元素之和，使得每行的和为1
a = a / torch.sum(a, 1, keepdim=True)

# 生成一个 3x2 的矩阵，其元素为0到9的随机整数，然后转为浮点数
b = torch.randint(0, 10, (3, 2)).float()

# 通过矩阵乘法计算矩阵 c，c 的每个元素是 a 的行与 b 的列的加权和
c = a @ b

# 输出矩阵 a
print('a=')
print(a)
print('--')

# 输出矩阵 b
print('b=')
print(b)
print('--')

# 输出矩阵 c
print('c=')
print(c)

print('-------------------')


B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
print(x.shape)
print('x=', x)