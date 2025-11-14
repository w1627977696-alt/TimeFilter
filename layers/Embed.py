"""
嵌入层模块
包含多种嵌入方法用于时间序列数据：
1. PositionalEmbedding: 位置编码
2. TokenEmbedding: token嵌入（通过1D卷积）
3. FixedEmbedding: 固定位置嵌入
4. TemporalEmbedding: 时间特征嵌入
5. TimeFeatureEmbedding: 时间特征嵌入（简化版）
6. DataEmbedding: 完整的数据嵌入（组合多种嵌入）
7. PatchEmbedding: Patch嵌入（用于PatchTST）
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from torch.nn.utils import weight_norm  # 权重归一化
import math  # 数学运算库


class PositionalEmbedding(nn.Module):
    """
    位置编码层
    使用正弦和余弦函数生成位置编码，用于捕获序列中的位置信息
    
    参数:
        d_model: 模型维度
        max_len: 最大序列长度
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 在对数空间中计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # 位置编码不需要梯度

        # 位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算分母项：10000^(2i/d_model)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 使用正弦函数编码偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数编码奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度
        pe = pe.unsqueeze(0)
        # 将位置编码注册为buffer（不是模型参数，但会被保存和加载）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        返回对应长度的位置编码
        
        输入:
            x: 输入张量，形状任意
        输出:
            位置编码，形状为 [1, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Token嵌入层
    使用1D卷积将输入特征映射到模型维度
    
    参数:
        c_in: 输入通道数（特征数）
        d_model: 模型维度
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 根据PyTorch版本选择padding值
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 1D卷积层：使用循环padding保持序列边界的连续性
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 使用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        前向传播
        
        输入:
            x: [B, T, C] - B: batch大小, T: 时间步长, C: 通道数
        输出:
            x: [B, T, d_model] - 嵌入后的特征
        """
        # 转置: [B, T, C] -> [B, C, T]
        # 卷积后转置回来: [B, d_model, T] -> [B, T, d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    固定位置嵌入层
    使用固定的正弦余弦位置编码（不可学习）
    
    参数:
        c_in: 输入维度（位置数量）
        d_model: 模型维度
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 创建位置编码权重
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False  # 固定权重，不需要梯度

        # 位置索引
        position = torch.arange(0, c_in).float().unsqueeze(1)
        # 计算分母项
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 正弦和余弦编码
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 创建嵌入层并设置为固定权重
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        返回固定的位置嵌入（detach防止梯度传播）
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    时间特征嵌入层
    对时间特征（月、日、周、小时、分钟）进行嵌入
    
    参数:
        d_model: 模型维度
        embed_type: 嵌入类型（'fixed' 或 'learned'）
        freq: 时间频率（'h': 小时, 't': 分钟等）
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # 定义各时间特征的取值范围
        minute_size = 4  # 分钟（15分钟间隔，共4个：0, 15, 30, 45）
        hour_size = 24  # 小时（0-23）
        weekday_size = 7  # 星期（0-6）
        day_size = 32  # 日期（1-31，加1个padding）
        month_size = 13  # 月份（1-12，加1个padding）

        # 根据嵌入类型选择固定或可学习的嵌入
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        # 如果频率是分钟级别，添加分钟嵌入
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        # 其他时间特征的嵌入
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        前向传播
        
        输入:
            x: [B, T, 4或5] - 时间特征（月、日、周、小时、[分钟]）
        输出:
            嵌入后的时间特征（所有时间特征的嵌入之和）
        """
        x = x.long()  # 转换为长整型
        
        # 获取分钟嵌入（如果存在）
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        # 获取其他时间特征的嵌入
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # 返回所有时间嵌入的总和
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    时间特征嵌入层（简化版）
    直接将时间特征线性映射到模型维度
    
    参数:
        d_model: 模型维度
        embed_type: 嵌入类型（这里总是'timeF'）
        freq: 时间频率
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # 根据频率映射到对应的特征数量
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]  # 输入维度
        # 线性映射层
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        前向传播：直接线性变换
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    完整的数据嵌入层
    组合了值嵌入、位置嵌入和时间嵌入
    
    参数:
        c_in: 输入通道数
        d_model: 模型维度
        embed_type: 嵌入类型
        freq: 时间频率
        dropout: Dropout比例
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 值嵌入：将原始数据映射到模型维度
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置嵌入：添加位置信息
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 时间嵌入：添加时间特征信息
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播
        
        输入:
            x: [B, T, C] - 输入数据
            x_mark: [B, T, time_dim] - 时间标记（可选）
        输出:
            嵌入后的特征
        """
        # 如果没有时间标记，只使用值嵌入和位置嵌入
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # 组合值嵌入、时间嵌入和位置嵌入
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    反转的数据嵌入层
    用于iTransformer等模型，将变量维度视为序列维度
    
    参数:
        c_in: 输入通道数（时间步长）
        d_model: 模型维度
        embed_type: 嵌入类型（未使用）
        freq: 时间频率（未使用）
        dropout: Dropout比例
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # 简单的线性映射（从时间步维度到模型维度）
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播
        
        输入:
            x: [B, T, C] - 输入数据
            x_mark: [B, T, time_dim] - 时间标记（可选）
        输出:
            嵌入后的特征 [B, C, d_model]
        """
        # 转置: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        
        # 如果没有时间标记
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 拼接时间标记后进行嵌入
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    不包含位置嵌入的数据嵌入层
    只使用值嵌入和时间嵌入
    
    参数:
        c_in: 输入通道数
        d_model: 模型维度
        embed_type: 嵌入类型
        freq: 时间频率
        dropout: Dropout比例
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播（不使用位置嵌入）
        
        输入:
            x: [B, T, C] - 输入数据
            x_mark: [B, T, time_dim] - 时间标记（可选）
        输出:
            嵌入后的特征
        """
        # 如果没有时间标记，只使用值嵌入
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 组合值嵌入和时间嵌入（不使用位置嵌入）
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch嵌入层
    用于PatchTST等基于patch的模型
    将时间序列分割成patch并嵌入到高维空间
    
    参数:
        d_model: 模型维度
        patch_len: Patch长度
        stride: Patch步长
        padding: 填充大小
        dropout: Dropout比例
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patch参数
        self.patch_len = patch_len
        self.stride = stride
        # 使用复制填充（在末尾复制最后一个值）
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone：将patch投影到d维向量空间
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # 位置嵌入
        self.position_embedding = PositionalEmbedding(d_model)

        # 残差dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        
        输入:
            x: [B, C, T] - B: batch大小, C: 通道数, T: 时间步长
        输出:
            x: [B*C, L, d_model] - L: patch数量
            n_vars: 变量数量
        """
        # 保存变量数量
        n_vars = x.shape[1]
        
        # 填充序列
        x = self.padding_patch_layer(x)
        
        # 使用unfold分割成patch
        # dimension=-1: 在最后一个维度（时间维度）上操作
        # size=self.patch_len: 每个patch的大小
        # step=self.stride: patch之间的步长
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 重塑: [B, C, L, P] -> [B*C, L, P]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # 输入编码：值嵌入 + 位置嵌入
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars
