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
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 训练模型
m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for step in range(10000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

cprint("### 4. 训练完成 ###", 'green')
print("最终损失: ", loss.item())

cprint("### 5. 生成的文本示例 ###", 'green')
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


