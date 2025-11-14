"""
TimeFilter 核心层实现文件
包含TimeFilter模型的核心组件：
1. GCN: 图卷积网络层
2. mask_moe: 混合专家模型的掩码生成器（动态路由）
3. GraphLearner: 自适应图学习器
4. GraphFilter: 图过滤层（结合图学习和图卷积）
5. GraphBlock: 完整的图处理块（包含图过滤和前馈网络）
6. TimeFilter_Backbone: TimeFilter的主干网络
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from torch.distributions.normal import Normal  # 正态分布（用于噪声注入）

class GCN(nn.Module):
    """
    图卷积网络（Graph Convolutional Network）
    对图结构数据进行卷积操作
    
    参数:
        dim: 特征维度
        n_heads: 多头数量
    """
    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)  # 特征投影层
        self.n_heads = n_heads  # 多头数量

    def forward(self, adj, x):
        """
        前向传播
        
        输入:
            adj: [B, H, L, L] - 邻接矩阵，B: batch大小, H: 头数, L: 节点数
            x: [B, L, D] - 节点特征，D: 特征维度
        输出:
            x: [B, L, D] - 图卷积后的节点特征
        """
        B, L, D = x.shape
        # 将特征投影并重塑为多头形式
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        # 对邻接矩阵进行归一化（每行和为1）
        adj = F.normalize(adj, p=1, dim=-1)
        # 执行图卷积：使用爱因斯坦求和约定进行矩阵乘法
        # "bhij,bjhd->bihd": 将邻接矩阵与节点特征相乘并聚合
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, L, H, D_]
        # 重塑回原始维度
        x = x.view(B, L, -1)
        return x

###############################
# 消融实验相关函数
###############################
def mask_topk_moe(adj, thre, n_vars, masks):
    """
    使用MoE（混合专家）阈值对邻接矩阵进行掩码
    将邻接矩阵分为三个区域（空间、时间、时空），并根据阈值过滤
    
    输入:
        adj: [B, H, L, L] - 邻接矩阵
        thre: [B, H, L, 3] - 三个区域的阈值
        n_vars: 变量数量
        masks: [L, 3, L] - 预定义的区域掩码
    输出:
        adj: [B, H, L, L] - 过滤后的邻接矩阵
    """
    # 如果掩码未提供，动态生成
    if masks is None:
        B, H, L, _ = adj.shape
        N = L // n_vars  # 每个变量的patch数量
        device = adj.device
        dtype = torch.float32
        print("Masks is None!")
        masks = []
        for k in range(L):
            # S: 空间掩码（相同时间位置的不同变量）
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            # T: 时间掩码（相同变量的不同时间位置）
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
            # ST: 时空掩码（其他所有位置）
            ST = torch.ones(L).to(dtype).to(device) - S - T
            masks.append(torch.stack([S, T, ST], dim=0))
        # [L, 3, L]
        masks = torch.stack(masks, dim=0)

    # 应用掩码到邻接矩阵的不同区域
    adj_mask0 = adj * masks[:,0,:]  # 空间区域
    adj_mask1 = adj * masks[:,1,:]  # 时间区域
    adj_mask2 = adj * masks[:,2,:]  # 时空区域

    # 根据阈值过滤每个区域
    adj_mask0[adj_mask0 <= thre[:, :, :, 0].unsqueeze(-1)] = 0
    adj_mask1[adj_mask1 <= thre[:, :, :, 1].unsqueeze(-1)] = 0 
    adj_mask2[adj_mask2 <= thre[:, :, :, 2].unsqueeze(-1)] = 0

    # 合并所有区域
    adj = adj_mask0 + adj_mask1 + adj_mask2

    return adj

def mask_topk_area(adj, n_vars, masks, alpha=0.5):
    """
    基于top-k过滤的区域掩码函数
    在每个区域内保留前alpha比例的最大值
    
    输入:
        adj: [B, H, L, L] - 邻接矩阵
        n_vars: 变量数量
        masks: [L, 3, L] - 预定义的区域掩码
        alpha: 保留的比例（0-1之间）
    输出:
        adj: [B, H, L, L] - 过滤后的邻接矩阵
    """
    # x: [B, H, L, L]
    B, H, L, _ = adj.shape
    N = L // n_vars  # 每个变量的patch数量
    
    # 如果掩码未提供，动态生成
    if masks is None:
        device = adj.device
        dtype = torch.float32
        print("Masks is None!")
        masks = []
        for k in range(L):
            # S: 空间掩码
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            # T: 时间掩码
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
            # ST: 时空掩码
            ST = torch.ones(L).to(dtype).to(device) - S - T
            masks.append(torch.stack([S, T, ST], dim=0))
        # [L, 3, L]
        masks = torch.stack(masks, dim=0)
    
    # 计算每个区域的元素数量
    n0 = n_vars - 1  # 空间区域的邻居数量
    n1 = N - 1  # 时间区域的邻居数量
    n2 = L - n0 - n1 - 1  # 时空区域的邻居数量

    # 应用掩码
    adj_mask0 = adj * masks[:,0,:]
    adj_mask1 = adj * masks[:,1,:]
    adj_mask2 = adj * masks[:,2,:]

    def apply_mask_to_region(adj_mask, n):
        """对单个区域应用top-k过滤"""
        threshold_idx = int(n * alpha)  # 计算阈值索引
        # 排序并获取阈值
        sorted_values, _ = torch.sort(adj_mask, dim=-1, descending=True)
        threshold = sorted_values[:, :, :, threshold_idx]
        # 保留大于等于阈值的值
        return adj_mask * (adj_mask >= threshold.unsqueeze(-1))

    # 对每个区域应用过滤
    adj_mask0 = apply_mask_to_region(adj_mask0, n0)
    adj_mask1 = apply_mask_to_region(adj_mask1, n1)
    adj_mask2 = apply_mask_to_region(adj_mask2, n2)

    # 合并所有区域
    adj = adj_mask0 + adj_mask1 + adj_mask2

    return adj
##########################

class mask_moe(nn.Module):
    """
    混合专家模型（Mixture of Experts, MoE）的掩码生成器
    使用门控机制动态选择要激活的图区域（空间、时间、时空）
    
    参数:
        n_vars: 变量数量
        top_p: 累积概率阈值（用于动态路由）
        num_experts: 专家数量（对应三个区域）
        in_dim: 输入维度
    """
    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts  # 专家数量（3个区域）
        self.n_vars = n_vars  # 变量数量
        self.in_dim = in_dim  # 输入维度

        # 门控网络：决定选择哪些专家
        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        # 噪声网络：为门控添加噪声以增加多样性
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True  # 是否使用噪声门控
        self.softplus = nn.Softplus()  # Softplus激活函数
        self.softmax = nn.Softmax(2)  # Softmax归一化
        self.top_p = top_p  # 累积概率阈值

    def cv_squared(self, x):
        """
        计算变异系数的平方
        用于衡量负载均衡性（专家使用的均匀程度）
        
        CV^2 = Var(x) / Mean(x)^2
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x):
        """
        计算交叉熵
        用于衡量专家选择的多样性
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        """
        带噪声的top-p门控机制
        
        输入:
            x: [B*H, L, L] - 输入特征（邻接矩阵）
            is_training: 是否处于训练模式
            noise_epsilon: 噪声的最小标准差
        输出:
            top_p_mask: [B, H, L, num_experts] - 选中的专家掩码
            loss: 负载均衡损失
        """
        # 计算干净的门控logits
        clean_logits = self.gate(x)
        
        # 如果在训练模式且启用噪声门控，添加噪声
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            # 计算噪声标准差
            noise_stddev = ((self.softplus(raw_noise) + noise_epsilon))
            # 添加高斯噪声
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # 将logits转换为概率
        logits = self.softmax(logits)
        # 计算动态选择的损失（鼓励多样性）
        loss_dynamic = self.cross_entropy(logits)

        # Top-p采样：选择累积概率超过top_p的专家
        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        # 确保至少选择一个专家
        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        # 创建top-p掩码
        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        # 将被过滤的概率设为0
        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        # 计算重要性损失（鼓励负载均衡）
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1  # 平衡两个损失的权重
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x, masks=None, is_training=None):
        """
        前向传播
        
        输入:
            x: [B, H, L, L] - 邻接矩阵
            masks: [L, 3, L] - 预定义的区域掩码（可选）
            is_training: 是否处于训练模式
        输出:
            mask: [B, H, L, L] - 组合后的掩码
            loss: 负载均衡损失
        """
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32

        # 创建基础掩码（对角线为1，防止自连接）
        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        
        # 如果top_p为0，不进行专家选择
        if self.top_p == 0.0:
            return mask_base, 0.0
        
        # 重塑输入以进行门控
        x = x.reshape(B * H, L, L)
        # 通过noisy top-k门控选择专家
        gates, loss = self.noisy_top_k_gating(x, is_training)
        gates = gates.reshape(B, H, L, -1).float()
        # [B, H, L, 3]

        # 如果未提供掩码，动态生成
        if masks is None:
            print("Masks is None!")
            masks = []
            N = L // self.n_vars
            for k in range(L):
                # S: 空间掩码
                S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
                # T: 时间掩码
                T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
                # ST: 时空掩码
                ST = torch.ones(L).to(dtype).to(device) - S - T
                masks.append(torch.stack([S, T, ST], dim=0))
            # [L, 3, L]
            masks = torch.stack(masks, dim=0)

        # 使用门控权重组合不同区域的掩码
        # "bhli,lid->bhld": 将门控权重与区域掩码相乘
        mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base

        return mask, loss


def mask_topk(x, alpha=0.5, largest=False):
    """
    Top-K掩码函数
    在邻接矩阵中保留每行的top-k个值（用于KNN图构建）
    
    输入:
        x: [B, H, L, L] - 邻接矩阵
        alpha: 保留的比例（0-1之间）
        largest: 是否保留最大的k个（True）或最小的k个（False）
    输出:
        mask: [B, H, L, L] - 掩码（1表示保留，0表示过滤）
    """
    # 计算要保留的元素数量
    k = int(alpha * x.shape[-1])
    # 获取top-k的索引
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    # 创建全1掩码
    mask = torch.ones_like(x, dtype=torch.float32)
    # 将top-k位置设为0（这里的逻辑看起来是反向的，实际上是用于后续的反向操作）
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask  # [B, H, L, L]


class GraphLearner(nn.Module):
    """
    自适应图学习器
    学习节点间的邻接矩阵（图结构）
    
    参数:
        dim: 特征维度
        n_vars: 变量数量
        top_p: MoE的top-p参数
        in_dim: 输入维度（用于MoE）
    """
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.dim = dim
        # 两个投影层用于计算节点间的相似度
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        # MoE掩码生成器
        self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        """
        前向传播
        
        输入:
            x: [B, H, L, D] - 多头节点特征
            masks: [L, 3, L] - 预定义的区域掩码（可选）
            alpha: KNN的保留比例
            is_training: 是否处于训练模式
        输出:
            adj: [B, H, L, L] - 学习到的邻接矩阵
            loss: MoE的负载均衡损失
        """
        # 计算邻接矩阵：通过两个投影层的内积
        # "bhid,bhjd->bhij": 计算节点i和节点j之间的相似度
        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))
        
        # 应用KNN过滤：只保留每个节点的top-k近邻
        adj = adj * mask_topk(adj, alpha)  # KNN
        
        # 通过MoE动态选择图区域
        mask, loss = self.mask_moe(adj, masks, is_training)
        
        # 应用MoE生成的掩码
        adj = adj * mask

        return adj, loss  # [B, H, L, L]


class GraphFilter(nn.Module):
    """
    图过滤层
    结合图学习和图卷积，对节点特征进行过滤和聚合
    
    参数:
        dim: 特征维度
        n_vars: 变量数量
        n_heads: 多头数量
        scale: 缩放因子
        top_p: MoE的top-p参数
        dropout: Dropout比例
        in_dim: 输入维度（用于MoE）
    """
    def __init__(self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        # 缩放因子：用于稳定训练
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        # 图学习器：学习邻接矩阵
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)
        # 图卷积层：在学习到的图上进行卷积
        self.graph_conv = GCN(self.dim, self.n_heads)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        """
        前向传播
        
        输入:
            x: [B, L, D] - 节点特征
            masks: [L, 3, L] - 预定义的区域掩码（可选）
            alpha: KNN的保留比例
            is_training: 是否处于训练模式
        输出:
            out: [B, L, D] - 过滤后的节点特征
            loss: MoE的负载均衡损失
        """
        B, L, D = x.shape

        # 将特征重塑为多头形式，并学习邻接矩阵
        adj, loss = self.graph_learner(
            x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3), 
            masks, 
            alpha, 
            is_training
        )  # [B, H, L, L]

        # 对邻接矩阵进行softmax归一化
        adj = torch.softmax(adj, dim=-1)
        # 应用dropout
        adj = self.dropout(adj)
        # 执行图卷积
        out = self.graph_conv(adj, x)
        
        return out, loss  # [B, L, D]


class GraphBlock(nn.Module):
    """
    图处理块
    包含图过滤层和前馈网络，以及残差连接和层归一化
    
    参数:
        dim: 特征维度
        n_vars: 变量数量
        d_ff: 前馈网络的隐藏维度
        n_heads: 多头数量
        top_p: MoE的top-p参数
        dropout: Dropout比例
        in_dim: 输入维度（用于MoE）
    """
    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        # 前馈网络维度，默认为dim的4倍
        self.d_ff = dim * 4 if d_ff is None else d_ff
        # 图过滤层（包含图学习和图卷积）
        self.gnn = GraphFilter(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(self.dim)
        # 前馈神经网络：两层全连接+GELU激活
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        """
        前向传播
        
        输入:
            x: [B, L, D] - 节点特征
            masks: [L, 3, L] - 预定义的区域掩码（可选）
            alpha: KNN的保留比例
            is_training: 是否处于训练模式
        输出:
            x: [B, L, D] - 处理后的节点特征
            loss: MoE的负载均衡损失
        """
        # 图过滤 + 残差连接（Pre-LN结构）
        out, loss = self.gnn(self.norm1(x), masks, alpha, is_training)
        x = x + out  # 残差连接
        
        # 前馈网络 + 残差连接（Pre-LN结构）
        x = x + self.ffn(self.norm2(x))
        
        return x, loss


class TimeFilter_Backbone(nn.Module):
    """
    TimeFilter主干网络
    由多个图处理块堆叠而成
    
    参数:
        hidden_dim: 隐藏维度
        n_vars: 变量数量
        d_ff: 前馈网络的隐藏维度
        n_heads: 多头数量
        n_blocks: 图块数量（网络深度）
        top_p: MoE的top-p参数
        dropout: Dropout比例
        in_dim: 输入维度（用于MoE）
    """
    def __init__(self, hidden_dim, n_vars, d_ff=None, n_heads=4, n_blocks=3, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = hidden_dim
        # 前馈网络维度，默认为隐藏维度的2倍
        self.d_ff = self.dim * 2 if d_ff is None else d_ff
        
        # 创建多个图处理块
        self.blocks = nn.ModuleList([
            GraphBlock(self.dim, n_vars, self.d_ff, n_heads, top_p, dropout, in_dim)
            for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        """
        前向传播
        
        输入:
            x: [B, N, T] - 输入特征
            masks: [L, 3, L] - 预定义的区域掩码（可选）
            alpha: KNN的保留比例
            is_training: 是否处于训练模式
        输出:
            x: [B, N, T] - 处理后的特征
            moe_loss: 所有块的平均MoE损失
        """
        # 初始化MoE损失
        moe_loss = 0.0
        
        # 依次通过每个图处理块
        for block in self.blocks:
            x, loss = block(x, masks, alpha, is_training)
            moe_loss += loss  # 累积损失
        
        # 计算平均损失
        moe_loss /= self.n_blocks
        
        return x, moe_loss  # [B, N, T]
