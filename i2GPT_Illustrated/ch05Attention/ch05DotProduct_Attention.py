#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch # 导入 torch
import torch.nn.functional as F # 导入 nn.functional
# 1. 创建两个张量 x1 和 x2
x1 = torch.randn(2, 3, 4) # 形状 (batch_size, seq_len1, feature_dim)
x2 = torch.randn(2, 5, 4) # 形状 (batch_size, seq_len2, feature_dim)
# 2. 计算原始权重
raw_weights = torch.bmm(x1, x2.transpose(1, 2)) # 形状 (batch_size, seq_len1, seq_len2)
# 3. 用 softmax 函数对原始权重进行归一化
attn_weights = F.softmax(raw_weights, dim=2) # 形状 (batch_size, seq_len1, seq_len2)
# 4. 将注意力权重与 x2 相乘，计算加权和
attn_output = torch.bmm(attn_weights, x2)  # 形状 (batch_size, seq_len1, feature_dim)
print('x1=', x1)
print('x2=', x2)
print('attn_output=', attn_output)
print('-'*20)
# In[13]:


# 创建两个张量 x1 和 x2
x1 = torch.randn(2, 3, 4) # 形状 (batch_size, seq_len1, feature_dim)
x2 = torch.randn(2, 5, 4) # 形状 (batch_size, seq_len2, feature_dim)
print("x1:", x1)
print("x2:", x2)


# In[14]:


# 计算点积，得到原始权重，形状为 (batch_size, seq_len1, seq_len2)
raw_weights = torch.bmm(x1, x2.transpose(1, 2))
print(" 原始权重：", raw_weights) 


# In[15]:


import torch.nn.functional as F # 导入 torch.nn.functional
# 应用 softmax 函数，使权重的值在 0 和 1 之间，且每一行的和为 1
attn_weights = F.softmax(raw_weights, dim=-1) # 归一化
print(" 归一化后的注意力权重：", attn_weights)


# In[16]:


# 与 x2 相乘，得到注意力分布的加权和，形状为 (batch_size, seq_len1, feature_dim)
attn_output = torch.bmm(attn_weights, x2)
print(" 注意力输出 :", attn_output)


# In[ ]:





# In[ ]:




