#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch # 导入 PyTorch 库
import random # 导入 random 库
import torch.nn as nn # 导入神经网络模块
import torch.optim as optim # 导入优化器模块


# In[9]:


# 构建一个非常简单的数据集
sentences = ["我 喜欢 玩具", "我 爱 爸爸", "我 讨厌 挨打"] 
# 将所有句子连接在一起，用空格分隔成多个词，再将重复的词去除，构建词汇表
word_list = list(set(" ".join(sentences).split())) 
# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)} 
# 创建一个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)} 
voc_size = len(word_list) # 计算词汇表的大小
print(' 词汇表：', word_to_idx) # 打印词汇到索引的映射字典
print(' 词汇表大小：', voc_size) # 打印词汇表大小


# In[10]:


# 构建批处理数据

batch_size = 2 # 每批数据的大小
def make_batch():
    input_batch = []  # 定义输入批处理列表
    target_batch = []  # 定义目标批处理列表
    selected_sentences = random.sample(sentences, batch_size) # 随机选择句子
    for sen in selected_sentences:  # 遍历每个句子
        word = sen.split()  # 用空格将句子分隔成多个词
        # 将除最后一个词以外的所有词的索引作为输入
        input = [word_to_idx[n] for n in word[:-1]]  # 创建输入数据
        # 将最后一个词的索引作为目标
        target = word_to_idx[word[-1]]  # 创建目标数据
        input_batch.append(input)  # 将输入添加到输入批处理列表
        target_batch.append(target)  # 将目标添加到目标批处理列表
    input_batch = torch.LongTensor(input_batch) # 将输入数据转换为张量
    target_batch = torch.LongTensor(target_batch) # 将目标数据转换为张量
    return input_batch, target_batch  # 返回输入批处理和目标批处理数据
input_batch, target_batch = make_batch() # 生成批处理数据
print(" 输入批处理数据：",input_batch)  # 打印输入批处理数据
# 将输入批处理数据中的每个索引值转换为对应的原始词
input_words = []
for input_idx in input_batch:
    input_words.append([idx_to_word[idx.item()] for idx in input_idx])
print(" 输入批处理数据对应的原始词：",input_words)
print(" 目标批处理数据：",target_batch) # 打印目标批处理数据
# 将目标批处理数据中的每个索引值转换为对应的原始词
target_words = [idx_to_word[idx.item()] for idx in target_batch]
print(" 目标批处理数据对应的原始词：",target_words)


# In[11]:


# 定义神经概率语言模型（NPLM）
class NPLM(nn.Module):
    def __init__(self):
        super(NPLM, self).__init__() 
        self.C = nn.Embedding(voc_size, embedding_size) # 定义一个词嵌入层
        print("初始化词嵌入层，大小为:", voc_size, embedding_size)
        # 第一个线性层，其输入大小为 n_step * embedding_size，输出大小为 n_hidden
        self.linear1 = nn.Linear(n_step * embedding_size, n_hidden) 
        print("初始化第一个线性层，输入大小:", n_step * embedding_size, "输出大小:", n_hidden)
        # 第二个线性层，其输入大小为 n_hidden，输出大小为 voc_size，即词汇表大小
        self.linear2 = nn.Linear(n_hidden, voc_size)
        print("初始化第二个线性层，输入大小:", n_hidden, "输出大小:", voc_size)


    def forward(self, X):
        global current_epoch, total_epochs
        # 判断是否在开始的10个或最后10个epoch
        if current_epoch < 3 or current_epoch >= total_epochs - 3:
            print('-'*25)
            print("\n原始输入的索引 (部分):", X[:, :2].numpy())
            X = self.C(X)
            print("经词嵌入层转换后的部分样本数据:", X[:, :2, :].detach().numpy())
            X = X.view(-1, n_step * embedding_size)
            print("重塑后的部分样本数据:", X[:2, :].detach().numpy())
            hidden = torch.tanh(self.linear1(X))
            print("第一个线性层和tanh后的部分样本数据:", hidden[:2, :].detach().numpy())
            output = self.linear2(hidden)
            print("最终输出层的部分样本数据:", output[:2, :].detach().numpy())
        else:
            X = self.C(X)
            X = X.view(-1, n_step * embedding_size)
            hidden = torch.tanh(self.linear1(X))
            output = self.linear2(hidden)
        return output


# In[7]:


n_step = 2 # 时间步数，表示每个输入序列的长度，也就是上下文长度 
n_hidden = 2 # 隐藏层大小
embedding_size = 2 # 词嵌入大小
model = NPLM() # 创建神经概率语言模型实例
print(' NPLM 模型结构：', model) # 打印模型的结构


# In[12]:
current_epoch = 0
total_epochs = 5000 

criterion = nn.CrossEntropyLoss() # 定义损失函数为交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.1) # 定义优化器为 Adam，学习率为 0.1
# 训练模型
for epoch in range(total_epochs):

    current_epoch = epoch
    optimizer.zero_grad() # 清除优化器的梯度
    input_batch, target_batch = make_batch() # 创建输入和目标批处理数据
    output = model(input_batch) # 将输入数据传入模型，得到输出结果
    loss = criterion(output, target_batch) # 计算损失值
    if (epoch + 1) % 1000 == 0: # 每 1000 次迭代，打印损失值
     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward() # 反向传播计算梯度
    optimizer.step() # 更新模型参数


# In[13]:


# 进行预测
input_strs = [['我', '讨厌'], ['我', '喜欢']]  # 需要预测的输入序列
# 将输入序列转换为对应的索引
input_indices = [[word_to_idx[word] for word in seq] for seq in input_strs]
# 将输入序列的索引转换为张量
input_batch = torch.LongTensor(input_indices) 
# 对输入序列进行预测，取输出中概率最大的类别
predict = model(input_batch).data.max(1)[1]  
# 将预测结果的索引转换为对应的词
predict_strs = [idx_to_word[n.item()] for n in predict.squeeze()]  
for input_seq, pred in zip(input_strs, predict_strs):
   print(input_seq, '->', pred)  # 打印输入序列和预测结果


# In[ ]:





# In[ ]:




