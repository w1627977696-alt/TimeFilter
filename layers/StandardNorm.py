"""
标准归一化模块
实现了可逆实例归一化（Reversible Instance Normalization, RevIN）
用于时间序列数据的归一化和反归一化操作
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块


class Normalize(nn.Module):
    """
    归一化层（支持可逆实例归一化）
    
    可以执行两种操作：
    1. 'norm': 归一化（减去均值，除以标准差）
    2. 'denorm': 反归一化（恢复原始尺度）
    
    参数:
        num_features: 特征数量（通道数）
        eps: 数值稳定性的小常数
        affine: 是否使用可学习的仿射参数（缩放和偏移）
        subtract_last: 是否减去最后一个值而非均值
        non_norm: 是否禁用归一化（直接返回输入）
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features  # 特征数量
        self.eps = eps  # 用于数值稳定性的小常数
        self.affine = affine  # 是否使用可学习参数
        self.subtract_last = subtract_last  # 是否使用最后一个值
        self.non_norm = non_norm  # 是否禁用归一化
        
        # 如果使用仿射变换，初始化参数
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        前向传播
        
        输入:
            x: 输入张量
            mode: 操作模式 ('norm' 或 'denorm')
        输出:
            归一化或反归一化后的张量
        """
        if mode == 'norm':
            # 归一化模式
            self._get_statistics(x)  # 计算统计量（均值和标准差）
            x = self._normalize(x)  # 执行归一化
        elif mode == 'denorm':
            # 反归一化模式
            x = self._denormalize(x)  # 恢复原始尺度
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """
        初始化可学习的仿射参数
        - affine_weight: 缩放参数，初始化为1
        - affine_bias: 偏移参数，初始化为0
        """
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        计算输入的统计量
        
        输入:
            x: 输入张量 [B, T, C]
        
        计算并保存：
        - mean或last: 用于中心化的值
        - stdev: 标准差
        """
        # 计算需要约简的维度（除了batch和特征维度）
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if self.subtract_last:
            # 使用最后一个时间步的值作为中心
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # 使用均值作为中心
            # detach()防止梯度传播到统计量
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        
        # 计算标准差
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        执行归一化操作
        
        步骤：
        1. 减去中心值（均值或最后一个值）
        2. 除以标准差
        3. 如果启用仿射变换，应用可学习的缩放和偏移
        """
        if self.non_norm:
            # 如果禁用归一化，直接返回
            return x
        
        # 步骤1: 中心化
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        
        # 步骤2: 标准化
        x = x / self.stdev
        
        # 步骤3: 仿射变换（如果启用）
        if self.affine:
            x = x * self.affine_weight  # 缩放
            x = x + self.affine_bias  # 偏移
        
        return x

    def _denormalize(self, x):
        """
        执行反归一化操作（恢复原始尺度）
        
        步骤：
        1. 如果使用了仿射变换，先反向应用
        2. 乘以标准差
        3. 加上中心值
        """
        if self.non_norm:
            # 如果禁用归一化，直接返回
            return x
        
        # 步骤1: 反向仿射变换（如果启用）
        if self.affine:
            x = x - self.affine_bias  # 移除偏移
            x = x / (self.affine_weight + self.eps * self.eps)  # 反向缩放
        
        # 步骤2: 反向标准化
        x = x * self.stdev
        
        # 步骤3: 反向中心化
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        
        return x
