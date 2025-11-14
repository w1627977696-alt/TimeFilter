"""
工具函数模块
包含训练过程中使用的各种工具函数：
- 学习率调整
- 早停机制
- 数据标准化
- 结果可视化
- 异常检测评估
"""

import os  # 操作系统接口

import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import matplotlib.pyplot as plt  # 绘图库
import pandas as pd  # 数据处理
import math  # 数学运算

plt.switch_backend('agg')  # 使用非交互式后端（适合服务器环境）


def adjust_learning_rate(optimizer, epoch, args):
    """
    调整学习率
    根据配置的学习率调整策略动态修改优化器的学习率
    
    参数:
        optimizer: PyTorch优化器
        epoch: 当前训练轮数
        args: 参数对象，包含学习率调整策略
    
    支持的策略:
        - type1: 每轮学习率减半
        - type2: 按预定义的轮数设置固定学习率
        - cosine: 余弦退火学习率
        - unchanged: 保持学习率不变
    """
    # 根据不同的学习率调整策略计算新的学习率
    if args.lradj == 'type1':
        # Type1: 指数衰减，每轮减半
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # Type2: 预定义的学习率schedule
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        # Cosine: 余弦退火，平滑地降低学习率
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'unchanged':
        # Unchanged: 保持学习率不变
        lr_adjust = {epoch: args.learning_rate}
    
    # 如果当前轮数在调整字典中，更新优化器的学习率
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    早停机制类
    当验证集性能不再提升时，提前停止训练以防止过拟合
    
    参数:
        patience: 容忍的最大epoch数（验证集性能不提升的轮数）
        verbose: 是否打印详细信息
        delta: 性能提升的最小阈值
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience  # 容忍轮数
        self.verbose = verbose  # 是否详细输出
        self.counter = 0  # 计数器：记录性能未提升的轮数
        self.best_score = None  # 最佳分数
        self.early_stop = False  # 是否应该早停
        self.val_loss_min = np.Inf  # 最小验证损失（初始化为无穷大）
        self.delta = delta  # 性能提升的最小阈值

    def __call__(self, val_loss, model, path):
        """
        判断是否需要早停并保存检查点
        
        参数:
            val_loss: 当前验证集损失
            model: 模型
            path: 模型保存路径
        """
        # 简化版本：每次都保存检查点（注释掉的代码是完整的早停逻辑）
        self.save_checkpoint(val_loss, model, path)
        '''
        # 完整的早停逻辑（已被注释）
        score = -val_loss  # 分数是负的损失（损失越小，分数越高）
        if self.best_score is None:
            # 第一次评估，保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # 性能没有提升（或提升小于delta）
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # 超过容忍次数，触发早停
                self.early_stop = True
        else:
            # 性能提升，保存模型并重置计数器
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        '''

    def save_checkpoint(self, val_loss, model, path):
        """
        保存模型检查点
        
        参数:
            val_loss: 验证集损失
            model: 模型
            path: 保存路径
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型状态字典
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        # 更新最小验证损失
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    点标记字典类
    允许使用点号访问字典属性（如dict.key而非dict['key']）
    """
    __getattr__ = dict.get  # 获取属性
    __setattr__ = dict.__setitem__  # 设置属性
    __delattr__ = dict.__delitem__  # 删除属性


class StandardScaler():
    """
    标准缩放器类
    用于数据的标准化和反标准化
    
    参数:
        mean: 数据均值
        std: 数据标准差
    """
    def __init__(self, mean, std):
        self.mean = mean  # 均值
        self.std = std  # 标准差

    def transform(self, data):
        """
        标准化数据
        
        公式: (data - mean) / std
        
        参数:
            data: 原始数据
        返回:
            标准化后的数据
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        反标准化数据（恢复原始尺度）
        
        公式: data * std + mean
        
        参数:
            data: 标准化后的数据
        返回:
            原始尺度的数据
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    可视化结果
    绘制真实值和预测值的对比图
    
    参数:
        true: 真实值
        preds: 预测值（可选）
        name: 保存图片的文件名
    """
    plt.figure()
    # 绘制真实值
    plt.plot(true, label='GroundTruth', linewidth=2)
    # 如果有预测值，一起绘制
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    # 保存图片
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    """
    调整异常检测结果
    使用point adjustment策略调整预测结果，
    如果预测到一个异常段中的任意一个点，则认为整个异常段都被检测到
    
    参数:
        gt: 真实标签（ground truth）
        pred: 预测标签
    返回:
        调整后的真实标签和预测标签
    """
    anomaly_state = False  # 当前是否处于异常状态
    for i in range(len(gt)):
        # 如果真实标签和预测都是异常，且之前不在异常状态
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # 向前填充：将该异常点之前的所有异常点的预测都设为1
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # 向后填充：将该异常点之后的所有异常点的预测都设为1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            # 不是异常，重置状态
            anomaly_state = False
        if anomaly_state:
            # 如果处于异常状态，将预测设为1
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    计算准确率
    
    参数:
        y_pred: 预测值
        y_true: 真实值
    返回:
        准确率（0-1之间）
    """
    return np.mean(y_pred == y_true)

