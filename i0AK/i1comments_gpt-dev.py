#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import torch
import torch.nn as nn
from torch.nn import functional as F
from termcolor import colored, cprint




# 读取文件内容
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
cprint("### 1. 数据集长度（字符数） ###", 'cyan')
print(len(text))
cprint("### 2. 数据集开头100个字符 ###", 'cyan')
print(text[:100])

# 获取所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
cprint("### 3. 唯一字符和字符集大小 ###", 'cyan')
print("唯一字符: ", ''.join(chars))
print("字符集大小: ", vocab_size)

# 创建字符到整数的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}




# 编码和解码函数
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)


# 划分数据集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1000)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y



# 定义模型
# class BigramLanguageModel(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

#     def forward(self, idx, targets=None):
#         logits = self.token_embedding_table(idx)
#         if targets is not None:
#             B, T, C = logits.shape
#             logits = logits.view(B * T, C)
#             targets = targets.view(B * T)
#             loss = F.cross_entropy(logits, targets)
#         else:
#             loss = None
#         return logits, loss

#     def generate(self, idx, max_new_tokens):
#         for _ in range(max_new_tokens):
#             logits, _ = self(idx)
#             logits = logits[:, -1, :]
#             probs = F.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probs, num_samples=1)
#             idx = torch.cat((idx, idx_next), dim=1)
#         return idx





class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        cprint("模型初始化，创建了一个大小为{}的词嵌入表。".format(vocab_size), 'yellow')
        print('token_embedding_table.weight=', self.token_embedding_table.weight)
        print('token_embedding_table.weight.shape=', self.token_embedding_table.weight.shape)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        cprint("#forward()前向传播过程：", 'blue')
        cprint("#forward()输入索引: {}".format(idx.tolist()), 'green')
        cprint("#forward()从词嵌入表中得到的 logits 形状: {}".format(logits.shape), 'green')
        cprint("#forward()logits: {}".format(logits), 'green')
        
        if targets is not None:
            B, T, C = logits.shape
            cprint("Batch size (B): {}".format(B), 'cyan')
            cprint("Time steps (T): {}".format(T), 'cyan')
            cprint("Features per time step (C): {}".format(C), 'cyan')

            cprint("原始 logits 形状: {}".format(logits.shape), 'magenta')
            cprint("原始 logits: \n{}".format(logits), 'magenta')
            cprint("原始 targets 形状: {}".format(targets.shape), 'magenta')
            cprint("原始 targets: \n{}".format(targets), 'magenta')

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            
            cprint("#forward()重塑后的 logits 形状: {}".format(logits.shape), 'green')
            cprint("#forward()重塑后的 logits: \n{}".format(logits), 'green')
            cprint("#forward()重塑后的 targets 形状: {}".format(targets.shape), 'green')
            cprint("#forward()重塑后的 targets: \n{}".format(targets), 'green')
            
            loss = F.cross_entropy(logits, targets)
            cprint("#forward()计算的损失: {:.4f}".format(loss.item()), 'red')
            return logits, loss
        else:
            return logits, None


    def generate(self, idx, max_new_tokens):
        cprint("@@generate() 开始文本生成过程...", 'red')
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            cprint("@@generate() 已生成的索引: {}".format(idx.tolist()), 'red')
        return idx

# 创建模型实例，使用更小的词汇表大小  
vocab_size = 5  
model = BigramLanguageModel(vocab_size)  
  
# 准备一个更小的批次的索引  
input_indices = torch.tensor([[1, 2]], dtype=torch.long)  
targets = torch.tensor([[2, 3]], dtype=torch.long)  
  
print('1   --------------------------->')  
# 执行前向传播
_, loss = model(input_indices, targets)
cprint("损失值: {:.4f}".format(loss.item()), 'red')

print('2   --------------------------->')
# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

# 训练模型  
for epoch in range(3):  # 假设我们训练3个周期  
    print(f"Epoch {epoch} - Before step: token_embedding_table weights = \n{model.token_embedding_table.weight}") 
    optimizer.zero_grad(set_to_none=True)  # 清除之前的梯度  
    _, loss = model(input_indices, targets)  # 执行前向传播和损失计算  
    loss.backward()  # 执行反向传播，计算梯度  
 
    optimizer.step()  # 更新模型参数  
    print(f"Epoch {epoch} - After step: token_embedding_table weights = \n{model.token_embedding_table.weight}")  
    cprint("Epoch {}: 损失值: {:.4f}".format(epoch, loss.item()), 'red')  

print('3   --------------------------->')
# 生成文本
start_idx = torch.tensor([[1]], dtype=torch.long)
generated_sequence = model.generate(start_idx, max_new_tokens=5)


print('------------------------------------------')
import time
time.sleep(3)

# # 训练模型
# m = BigramLanguageModel(vocab_size)
# optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# batch_size = 32
# for step in range(10000):
#     xb, yb = get_batch('train')
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# cprint("### 4. 训练完成 ###", 'green')
# print("最终损失: ", loss.item())

# cprint("### 5. 生成的文本示例 ###", 'green')
# print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


