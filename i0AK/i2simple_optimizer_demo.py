import torch  
import torch.nn as nn  
import torch.optim as optim  
import matplotlib.pyplot as plt  
  
# 简单的线性回归模型  
model = nn.Linear(1, 1)  
  
# 使用较小的学习率  
optimizer = optim.SGD(model.parameters(), lr=0.001)  
  
# 损失函数  
criterion = nn.MSELoss()  
  
# 创建一些简单的数据  
X = torch.tensor([[1.0], [2.0], [3.0]])  
Y = torch.tensor([[5.0], [7.0], [9.0]])  
  
# 用于绘图的数据  
losses = []  
  
# 训练模型  
for epoch in range(200):  # 增加训练轮数  
    optimizer.zero_grad()  # 清除旧的梯度  
    outputs = model(X)  # 前向传播  
    loss = criterion(outputs, Y)  # 计算损失  
    loss.backward()  # 反向传播  
    optimizer.step()  # 更新参数  
      
    # 记录损失  
    losses.append(loss.item())  
      
    # 打印模型参数  
    # print(f"Epoch {epoch+1}/200")  
    # for name, param in model.named_parameters():  
    #     print(f"{name}: {param.data}")  
  
# 可视化损失下降  
plt.figure(figsize=(10, 6))  
plt.plot(losses)  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.title('Loss During Training')  
plt.grid(True)  
plt.show()  
  
# 新的输入数据  
new_X = torch.tensor([[5.0], [10.0], [15.0]])  
  
# 使用模型进行预测  
with torch.no_grad():  # 确保不计算梯度  
    new_predictions = model(new_X)  
  
# 打印预测结果  
print("Predictions:", new_predictions)  
