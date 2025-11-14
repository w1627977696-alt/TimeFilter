"""
评估指标模块
包含各种用于评估时间序列预测性能的指标函数
"""

import numpy as np  # 数值计算库


def RSE(pred, true):
    """
    相对平方误差 (Relative Squared Error)
    衡量预测值与真实值之间的相对误差
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        RSE值
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    相关系数 (Correlation)
    衡量预测值与真实值之间的线性相关性
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        相关系数（-1到1之间）
    """
    # 计算分子：中心化后的预测值和真实值的乘积之和
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    # 计算分母：两者标准差的乘积
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    平均绝对误差 (Mean Absolute Error)
    衡量预测值与真实值之间的平均绝对差异
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        MAE值（越小越好）
    """
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    """
    均方误差 (Mean Squared Error)
    衡量预测值与真实值之间的平均平方差异
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        MSE值（越小越好）
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    """
    均方根误差 (Root Mean Squared Error)
    MSE的平方根，与原始数据具有相同的单位
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        RMSE值（越小越好）
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    平均绝对百分比误差 (Mean Absolute Percentage Error)
    衡量预测值与真实值之间的平均百分比误差
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        MAPE值（越小越好）
    """
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    """
    均方百分比误差 (Mean Squared Percentage Error)
    衡量预测值与真实值之间的平均平方百分比误差
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        MSPE值（越小越好）
    """
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    """
    计算所有评估指标
    一次性计算多个常用指标
    
    参数:
        pred: 预测值
        true: 真实值
    返回:
        mae: 平均绝对误差
        mse: 均方误差
        rmse: 均方根误差
        mape: 平均绝对百分比误差
        mspe: 均方百分比误差
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
