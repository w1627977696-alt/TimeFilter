"""
数据工厂模块
提供统一的数据加载接口，根据数据集类型返回相应的数据加载器
"""

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, \
                                      Dataset_PEMS, Dataset_Climate  # 导入各种数据集类
from data_provider.uea import collate_fn  # UEA数据集的批处理函数
from torch.utils.data import DataLoader  # PyTorch数据加载器
import torch

# 数据集字典：将数据集名称映射到对应的数据集类
data_dict = {
    'ETTh1': Dataset_ETT_hour,  # ETT小时级数据集1
    'ETTh2': Dataset_ETT_hour,  # ETT小时级数据集2
    'ETTm1': Dataset_ETT_minute,  # ETT分钟级数据集1
    'ETTm2': Dataset_ETT_minute,  # ETT分钟级数据集2
    'custom': Dataset_Custom,  # 自定义数据集
    'Solar': Dataset_Solar,  # 太阳能数据集
    'PEMS': Dataset_PEMS,  # PEMS交通数据集
    'Climate': Dataset_Climate,  # 气候数据集
}


def data_provider(args, flag):
    """
    数据提供函数
    根据配置和标志创建数据集和数据加载器
    
    参数:
        args: 命令行参数对象，包含数据配置
        flag: 数据集类型标志 ('train', 'val', 'test')
    
    返回:
        data_set: 数据集对象
        data_loader: 数据加载器对象
    """
    # 根据数据集名称获取对应的数据集类
    Data = data_dict[args.data]
    # 根据嵌入类型选择时间编码方式
    timeenc = 0 if args.embed != 'timeF' else 1

    # 根据标志设置是否打乱数据
    shuffle_flag = False if flag == 'test' else True  # 测试集不打乱，训练集和验证集打乱
    drop_last = False  # 不丢弃最后一个不完整的batch
    batch_size = args.batch_size  # 批次大小
    freq = args.freq  # 时间频率

    # 创建数据集实例
    data_set = Data(
        args = args,  # 传递所有参数
        root_path=args.root_path,  # 数据根路径
        data_path=args.data_path,  # 数据文件路径
        flag=flag,  # 数据集类型（train/val/test）
        size=[args.seq_len, args.label_len, args.pred_len],  # 序列长度配置
        features=args.features,  # 特征类型（M/S/MS）
        target=args.target,  # 目标特征
        timeenc=timeenc,  # 时间编码方式
        freq=freq,  # 时间频率
        seasonal_patterns=args.seasonal_patterns  # 季节性模式（用于M4数据集）
    )
    print(flag, len(data_set))  # 打印数据集大小
    
    # 创建数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,  # 批次大小
        shuffle=shuffle_flag,  # 是否打乱
        num_workers=args.num_workers,  # 数据加载的工作进程数
        drop_last=drop_last  # 是否丢弃最后一个不完整的batch
    )

    return data_set, data_loader
