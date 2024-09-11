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


  
B, T, C = 4, 8, 2 # 批次数，时间步长，通道数  
x = torch.randn(B, T, C)  # 创建一个随机的张量  
  
print("原始张量 x 的形状:", x.shape)  
print("x =", x)  
  
xbow = torch.zeros((B, T, C))  # 创建一个形状相同的零张量，用于存储计算结果  
  
for b in range(B):  # 遍历每一个批次  
    for t in range(T):  # 遍历每一个时间步  
        xprev = x[b, :t+1]  # 取当前批次从时间开始到当前时间步的所有数据  
        xbow[b, t] = torch.mean(xprev, 0)  # 计算平均值并存储  
          
        # 打印当前批次和时间步的信息  
        print(f"批次 {b+1}, 时间步 {t+1} 的数据 xprev:")  
        print(xprev)  
        print(f"批次 {b+1}, 时间步 {t+1} 的平均值:")  
        print(xbow[b, t])  
  
# 最终的 xbow 输出  
print("最终的累积平均值张量 xbow:")  
print(xbow)  
