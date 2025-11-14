"""
TimeFilter 模型定义文件
实现了TimeFilter模型的主要结构，包括：
1. PatchEmbed: 将时间序列分割成patch并进行嵌入
2. Model: TimeFilter的完整模型结构
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
import math  # 数学运算库

from layers.Embed import PositionalEmbedding  # 位置编码层
from layers.StandardNorm import Normalize  # 标准化/归一化层
from layers.TimeFilter_layers import TimeFilter_Backbone  # TimeFilter的主干网络


class PatchEmbed(nn.Module):
    """
    Patch嵌入层
    将时间序列分割成固定长度的patch，并投影到高维空间
    
    参数:
        dim: 嵌入维度（输出维度）
        patch_len: 每个patch的长度
        stride: patch之间的步长，默认等于patch_len（无重叠）
        pos: 是否使用位置编码
    """
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len  # 保存patch长度
        self.stride = patch_len if stride is None else stride  # 设置步长，默认等于patch长度
        self.patch_proj = nn.Linear(self.patch_len, dim)  # 线性投影层：将patch映射到dim维
        self.pos = pos  # 是否使用位置编码
        if self.pos:
            pos_emb_theta = 10000  # 位置编码的theta参数
            self.pe = PositionalEmbedding(dim, pos_emb_theta)  # 创建位置编码层
    
    def forward(self, x):
        """
        前向传播
        
        输入:
            x: [B, N, T] - B: batch大小, N: 变量数量, T: 时间步长
        输出:
            x: [B, N*L, D] - L: patch数量, D: 嵌入维度
        """
        # 使用unfold将时间序列分割成patch
        # dimension=-1: 在最后一个维度（时间维度）上操作
        # size=self.patch_len: 每个patch的大小
        # step=self.stride: patch之间的步长
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P] - P是patch长度
        
        # 通过线性层将每个patch投影到dim维空间
        x = self.patch_proj(x)  # [B, N*L, D]
        
        # 如果启用位置编码，添加位置信息
        if self.pos:
            x += self.pe(x)  # 加上位置编码
        
        return x

class Model(nn.Module):
    """
    TimeFilter 主模型
    实现了基于patch的时空图过滤的时间序列预测模型
    
    模型流程：
    1. 输入归一化
    2. 将时间序列分割成patch并嵌入
    3. 通过TimeFilter主干网络处理（包含图学习和图卷积）
    4. 投影回预测长度
    5. 输出反归一化
    
    参数:
        configs: 配置对象，包含所有超参数
    """
    def __init__(self, configs):
        super().__init__()

        # ========== 基础配置 ==========
        self.task_name = configs.task_name  # 任务名称
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.n_vars = configs.c_out  # 变量数量（通道数）
        self.dim = configs.d_model  # 模型隐藏维度
        self.d_ff = configs.d_ff  # 前馈网络维度
        self.patch_len = configs.patch_len  # patch长度
        self.stride = self.patch_len  # patch步长（无重叠）
        # 计算patch数量：L = (T - P) / S + 1
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # L

        # ========== 图过滤参数 ==========
        # alpha: KNN图构建参数，控制保留的近邻比例
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        # top_p: MoE动态路由参数，控制专家选择的累积概率阈值
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # ========== 模型组件 ==========
        # Patch嵌入层：将时间序列分割成patch并嵌入到高维空间
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter主干网络：包含多层图块（GraphBlock）
        # 每个图块包含图学习器、图卷积和前馈网络
        self.backbone = TimeFilter_Backbone(
            self.dim,  # 隐藏维度
            self.n_vars,  # 变量数量
            self.d_ff,  # 前馈网络维度
            configs.n_heads,  # 注意力头数
            configs.e_layers,  # 编码器层数（图块数量）
            self.top_p,  # MoE的top-p参数
            configs.dropout,  # Dropout比例
            self.seq_len * self.n_vars // self.patch_len  # 总patch数量
        )
        
        # 预测头：将patch表示映射回预测长度
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        # ========== 归一化层 ==========
        # 不使用可逆实例归一化（RevIN），使用标准归一化
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)
    
    def forward(self, x, masks, is_training=False, target=None):
        """
        前向传播
        
        输入:
            x: [B, T, C] - B: batch大小, T: 时间步长, C: 通道数
            masks: 预计算的掩码，用于定义不同的图区域（空间、时间、时空）
            is_training: 是否处于训练模式
            target: 目标值（可选，用于某些任务）
        
        输出:
            x: [B, pred_len, C] - 预测结果
            moe_loss: 混合专家模型的辅助损失
        """
        # x: [B, T, C]
        B, T, C = x.shape
        
        # ========== 步骤1: 归一化 ==========
        # 对输入进行标准化，减去均值并除以标准差
        x = self.norm(x, 'norm')
        
        # ========== 步骤2: 重塑并嵌入 ==========
        # 将 [B, T, C] 转换为 [B, C, T]
        x = x.permute(0, 2, 1).reshape(-1, C*T)  # [B, C*T]
        # 通过patch嵌入层，将序列分割成patch并嵌入
        x = self.patch_embed(x)  # [B, N, D]  N = [C*T / P] = C * L

        # ========== 步骤3: 主干网络处理 ==========
        # 通过TimeFilter主干网络，进行图学习和图卷积
        # moe_loss: 混合专家模型的负载均衡损失
        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)

        # ========== 步骤4: 预测头 ==========
        # 重塑: [B, N, D] -> [B, C, L, D]
        # 展平: [B, C, L, D] -> [B, C, L*D]
        # 线性投影: [B, C, L*D] -> [B, C, pred_len]
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))
        
        # 转置: [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)
        
        # ========== 步骤5: 反归一化 ==========
        # 将预测结果转换回原始尺度
        x = self.norm(x, 'denorm')

        return x, moe_loss
        