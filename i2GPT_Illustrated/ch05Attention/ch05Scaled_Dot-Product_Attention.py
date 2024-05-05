import torch  # 导入 torch
import torch.nn.functional as F  # 导入 nn.functional

# 1. 创建两个张量 x1 和 x2
x1 = torch.randn(2, 3, 4)  # 形状 (batch_size, seq_len1, feature_dim)
x2 = torch.randn(2, 5, 4)  # 形状 (batch_size, seq_len2, feature_dim)
# 2. 计算张量点积，得到原始权重
raw_weights = torch.bmm(x1, x2.transpose(1, 2))  # 形状 (batch_size, seq_len1, seq_len2)
print("raw_weights=", raw_weights)
# 3. 将原始权重除以缩放因子
scaling_factor = x1.size(-1) ** 0.5
scaled_weights = raw_weights / scaling_factor  # 形状 (batch_size, seq_len1, seq_len2)
print("scaled_weights=", scaled_weights)
# 4. 对原始权重进行归一化
attn_weights = F.softmax(
    scaled_weights, dim=2
)  #  形 状 (batch_size,  seq_len1,  seq_len2)

# 5. 使用注意力权重对 x2 加权求和
attn_output = torch.bmm(attn_weights, x2)  # 形状 (batch_size, seq_len1, feature_dim)
print("attn_output=", attn_output)
