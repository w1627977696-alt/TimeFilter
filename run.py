"""
TimeFilter 时间序列预测模型的主入口文件
这个文件负责设置命令行参数、初始化模型以及执行训练和测试流程
"""

import argparse  # 用于解析命令行参数
import os  # 操作系统接口，用于文件路径操作
import torch  # PyTorch深度学习框架
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # 长期预测实验类
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast  # 短期预测实验类
from utils.print_args import print_args  # 打印参数的工具函数
import random  # 随机数生成器
import numpy as np  # 数值计算库

if __name__ == '__main__':
    # 设置固定的随机种子以确保实验的可重复性
    fix_seed = 2021
    random.seed(fix_seed)  # 设置Python随机种子
    torch.manual_seed(fix_seed)  # 设置PyTorch随机种子
    np.random.seed(fix_seed)  # 设置NumPy随机种子

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='PatchTST')

    # ==================== 基础配置 ====================
    # 任务名称：长期预测、短期预测、插补、分类或异常检测
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='任务名称，可选项：[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # 是否处于训练状态：1表示训练，0表示测试
    parser.add_argument('--is_training', type=int, required=True, default=1, help='训练/测试状态标志')
    # 模型标识符，用于区分不同的实验
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型ID')
    # 模型名称
    parser.add_argument('--model', type=str, required=True, default='PatchTST',
                        help='模型名称，可选项：[Autoformer, Transformer, TimesNet]')

    # ==================== 数据加载器配置 ====================
    # 数据集类型
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='数据集类型')
    # 数据文件的根路径
    parser.add_argument('--root_path', type=str, default='./data/', help='数据文件的根目录路径')
    # 数据文件名
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名')
    # 预测任务特征类型：M=多变量预测多变量，S=单变量预测单变量，MS=多变量预测单变量
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型，可选项：[M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
    # 在S或MS任务中的目标特征
    parser.add_argument('--target', type=str, default='OT', help='S或MS任务中的目标特征列名')
    # 时间特征编码的频率
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码频率，可选项：[s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月]，也可以使用更详细的频率如15min或3h')
    # 模型检查点保存位置
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的保存位置')

    # ==================== 预测任务配置 ====================
    # 输入序列长度（历史窗口大小）
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    # 起始令牌长度（用于decoder的起始部分）
    parser.add_argument('--label_len', type=int, default=48, help='起始令牌长度')
    # 预测序列长度（预测未来的时间步数）
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    # M4数据集的季节性模式
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4数据集的子集类型')
    # 是否对输出数据进行逆变换
    parser.add_argument('--inverse', action='store_true', help='是否对输出数据进行逆变换', default=False)

    # ==================== 插补任务配置 ====================
    # 遮蔽比例（用于数据插补任务）
    parser.add_argument('--mask_rate', type=float, default=0.25, help='遮蔽比例')

    # ==================== 异常检测任务配置 ====================
    # 先验异常比例
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='先验异常比例（百分比）')

    # ==================== 模型定义配置 ====================
    # TimesBlock的top-k参数
    parser.add_argument('--top_k', type=int, default=5, help='TimesBlock的top-k参数')
    # Inception的卷积核数量
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception的卷积核数量')
    # 编码器输入维度（输入变量的数量）
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入维度')
    # 解码器输入维度
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入维度')
    # 输出维度（输出变量的数量）
    parser.add_argument('--c_out', type=int, default=7, help='输出维度')
    # 模型的隐藏维度
    parser.add_argument('--d_model', type=int, default=512, help='模型的隐藏维度')
    # 注意力头的数量
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头的数量')
    # 编码器层数
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    # 解码器层数
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    # 前馈神经网络的维度
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈神经网络的维度')
    # 移动平均窗口大小
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均的窗口大小')
    # 注意力因子
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    # 位置编码：0表示不使用，1表示使用
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='位置编码。设置pos为0或1')
    # 是否在编码器中使用蒸馏
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏，使用此参数表示不使用蒸馏',
                        default=True)
    # Dropout比例
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
    # 时间特征编码方式
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式，可选项：[timeF, fixed, learned]')
    # 激活函数类型
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数类型')
    # 是否在编码器中输出注意力权重
    parser.add_argument('--output_attention', action='store_true', help='是否在编码器中输出注意力权重')
    # 通道独立性：0表示通道相关，1表示通道独立（用于FreTS模型）
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: 通道相关 1: 通道独立（用于FreTS模型）')
    # 序列分解方法
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='序列分解方法，仅支持moving_avg或dft_decomp')
    # 是否使用归一化：1表示使用，0表示不使用
    parser.add_argument('--use_norm', type=int, default=1, help='是否使用归一化；True为1，False为0')
    # 下采样层数
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='下采样层数')
    # 下采样窗口大小
    parser.add_argument('--down_sampling_window', type=int, default=1, help='下采样窗口大小')
    # 下采样方法
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='下采样方法，仅支持avg、max或conv')
    
    # ==================== TimeFilter 特定参数 ====================
    # patch的长度（将时间序列分割成patch的长度）
    parser.add_argument('--patch_len', type=int, default=16, help='patch的长度')
    # KNN图构建的alpha参数（控制保留的近邻数量）
    parser.add_argument('--alpha', type=float, default=0.1, help='用于图构建的KNN参数')
    # MoE（混合专家）中的动态路由top-p参数
    parser.add_argument('--top_p', type=float, default=0.5, help='混合专家模型中的动态路由参数')

    # ==================== 优化器配置 ====================
    # 数据加载器的工作进程数
    parser.add_argument('--num_workers', type=int, default=1, help='数据加载器的工作进程数')
    # 实验重复次数
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    # 训练轮数
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    # 批次大小
    parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批次大小')
    # 早停耐心值（验证集性能多少轮不提升后停止训练）
    parser.add_argument('--patience', type=int, default=3, help='早停的耐心值')
    # 优化器学习率
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    # 实验描述
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    # 损失函数类型
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数类型')
    # 学习率调整策略
    parser.add_argument('--lradj', type=str, default='cosine', help='学习率调整策略')
    # 是否使用自动混合精度训练
    parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度训练', default=False)

    # ==================== GPU配置 ====================
    # 是否使用GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    # GPU设备编号
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备编号')
    # 是否使用多GPU
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多个GPU', default=False)
    # 多GPU的设备ID列表
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU的设备ID列表')

    # ==================== 去平稳化投影器参数 ====================
    # 投影器的隐藏层维度列表
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影器的隐藏层维度列表')
    # 投影器的隐藏层数量
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='投影器的隐藏层数量')

    # ==================== 评估指标配置（DTW） ====================
    # 是否使用DTW指标（DTW计算耗时，除非必要否则不建议使用）
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='是否使用DTW指标（DTW计算耗时，除非必要否则不建议使用）')
    
    # ==================== 数据增强配置 ====================
    # 数据增强的倍数（0表示不进行数据增强）
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="数据增强的倍数")
    # 随机化种子
    parser.add_argument('--seed', type=int, default=2, help="随机化种子")
    # 抖动增强：在时间序列中添加随机噪声
    parser.add_argument('--jitter', default=False, action="store_true", help="抖动预设增强")
    # 缩放增强：对时间序列进行缩放
    parser.add_argument('--scaling', default=False, action="store_true", help="缩放预设增强")
    # 等长排列增强：对固定长度的序列段进行排列
    parser.add_argument('--permutation', default=False, action="store_true", help="等长排列预设增强")
    # 随机长度排列增强：对随机长度的序列段进行排列
    parser.add_argument('--randompermutation', default=False, action="store_true", help="随机长度排列预设增强")
    # 幅度扭曲增强：对时间序列的幅度进行扭曲
    parser.add_argument('--magwarp', default=False, action="store_true", help="幅度扭曲预设增强")
    # 时间扭曲增强：对时间轴进行扭曲
    parser.add_argument('--timewarp', default=False, action="store_true", help="时间扭曲预设增强")
    # 窗口切片增强：从时间序列中切片窗口
    parser.add_argument('--windowslice', default=False, action="store_true", help="窗口切片预设增强")
    # 窗口扭曲增强：对窗口进行扭曲
    parser.add_argument('--windowwarp', default=False, action="store_true", help="窗口扭曲预设增强")
    # 旋转增强：旋转时间序列
    parser.add_argument('--rotation', default=False, action="store_true", help="旋转预设增强")
    # SPAWNER增强：基于SPAWNER算法的增强
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER预设增强")
    # DTW扭曲增强：基于DTW的扭曲增强
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW扭曲预设增强")
    # Shape DTW扭曲增强：基于Shape DTW的扭曲增强
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW扭曲预设增强")
    # 加权DBA增强：基于加权DBA算法的增强
    parser.add_argument('--wdba', default=False, action="store_true", help="加权DBA预设增强")
    # 判别式DTW扭曲增强：基于判别式DTW的扭曲增强
    parser.add_argument('--discdtw', default=False, action="store_true", help="判别式DTW扭曲预设增强")
    # 判别式Shape DTW扭曲增强：基于判别式Shape DTW的扭曲增强
    parser.add_argument('--discsdtw', default=False, action="store_true", help="判别式shapeDTW扭曲预设增强")
    # 额外标签
    parser.add_argument('--extra_tag', type=str, default="", help="额外标签")

    # 解析命令行参数
    args = parser.parse_args()
    # 检查GPU是否可用，并相应地设置use_gpu标志
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 如果使用多GPU，解析设备ID
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')  # 移除空格
        device_ids = args.devices.split(',')  # 按逗号分割设备ID
        args.device_ids = [int(id_) for id_ in device_ids]  # 转换为整数列表
        args.gpu = args.device_ids[0]  # 将第一个GPU设为主GPU

    # 打印实验参数
    print('Args in experiment:')
    print_args(args)

    # 根据任务名称选择相应的实验类
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast  # 长期预测实验
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast  # 短期预测实验
    else:
        Exp = Exp_Long_Term_Forecast  # 默认使用长期预测

    # 如果是训练模式
    if args.is_training:
        # 进行多次实验迭代
        for ii in range(args.itr):
            # 创建实验实例
            exp = Exp(args)
            
            # 生成实验设置字符串，用于标识和保存实验结果
            # 格式：任务名_模型ID_模型名_数据集_特征类型_序列长度_标签长度_预测长度_
            #      模型维度_注意力头数_编码器层数_解码器层数_前馈网络维度_因子_嵌入类型_蒸馏_描述_迭代次数
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            # 开始训练
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # 训练完成后进行测试
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            # 清空GPU缓存以释放内存
            torch.cuda.empty_cache()
    else:
        # 如果是测试模式（不训练）
        ii = 0  # 迭代次数设为0
        
        # 生成实验设置字符串
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        # 创建实验实例
        exp = Exp(args)
        
        # 仅进行测试（加载预训练模型）
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)  # test=1表示加载已保存的模型检查点
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
