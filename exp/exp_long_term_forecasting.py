"""
长期预测实验类
实现TimeFilter模型的长期时间序列预测任务
包含完整的训练、验证和测试流程，以及数据维度的详细说明
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    长期预测实验类
    
    继承自Exp_Basic，专门用于长期时间序列预测任务
    主要功能：
    1. 构建TimeFilter模型
    2. 生成图过滤所需的掩码
    3. 执行训练、验证和测试流程
    4. 计算和保存预测结果
    
    参数:
        args: 命令行参数对象
    """
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # 预先生成掩码，用于TimeFilter的图学习
        # masks形状: [L, 3, L] 其中L = seq_len * c_out // patch_len
        self.masks = self._get_mask()

    def _build_model(self):
        """
        构建模型
        
        返回:
            model: TimeFilter模型实例
        """
        # 从模型字典中获取对应的模型类并实例化
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU，将模型包装为DataParallel
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        获取数据集和数据加载器
        
        参数:
            flag: 数据集类型标志 ('train', 'val', 'test')
        
        返回:
            data_set: 数据集对象
            data_loader: 数据加载器对象
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        选择优化器
        
        返回:
            model_optim: Adam优化器
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择损失函数
        
        返回:
            criterion: MSE损失函数
        """
        criterion = nn.MSELoss()
        return criterion
    
    def _get_mask(self):
        """
        生成图过滤的掩码
        
        掩码用于定义三种不同的图连接区域：
        1. S (Spatial): 空间掩码 - 同一时间位置的不同变量之间的连接
        2. T (Temporal): 时间掩码 - 同一变量的不同时间位置之间的连接
        3. ST (Spatio-Temporal): 时空掩码 - 其他所有跨变量跨时间的连接
        
        维度说明：
        - L: 总的patch数量 = seq_len * c_out // patch_len
        - N: 每个变量的patch数量 = seq_len // patch_len
        - c_out: 变量数量（通道数）
        
        返回:
            masks: [L, 3, L] 形状的掩码张量
                - 第一维：对应每个patch位置
                - 第二维：对应三种区域类型 [S, T, ST]
                - 第三维：对应目标patch位置
        """
        dtype = torch.float32
        # L: 总patch数 = 变量数 * 每个变量的patch数
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        # N: 每个变量的patch数
        N = self.args.seq_len // self.args.patch_len
        masks = []
        
        # 为每个patch位置k生成三种区域的掩码
        for k in range(L):
            # S掩码: 空间连接 - 相同时间位置(k % N)的不同变量
            # 条件: 时间位置相同 且 不是自己
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            
            # T掩码: 时间连接 - 相同变量(k // N)的不同时间位置
            # 条件: 在同一变量的范围内 且 不是自己
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            
            # ST掩码: 时空连接 - 所有其他连接（既不是S也不是T）
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0  # 排除自连接
            
            # 堆叠三种掩码: [3, L]
            masks.append(torch.stack([S, T, ST], dim=0))
        
        # 堆叠所有位置的掩码: [L, 3, L]
        masks = torch.stack(masks, dim=0)
        return masks
    
    def _get_mask_2(self):
        """
        生成图过滤掩码的替代方法（未使用）
        
        使用不同的实现方式生成相同的三种区域掩码
        
        返回:
            masks: [3, 1, L, L] 形状的掩码张量
        """
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len

        # 基础掩码：单位矩阵（用于排除自连接）
        mask_base = torch.eye(L, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        
        # mask0: 空间掩码
        mask0 = torch.eye(L, device=self.device, dtype=dtype)
        mask0.view(self.args.c_out, N, self.args.c_out, N).diagonal(dim1=0, dim2=2).fill_(1)
        mask0 = mask0.unsqueeze(0).unsqueeze(0) - mask_base
        
        # mask1: 时间掩码（使用Kronecker积构建）
        mask1 = torch.kron(torch.ones(self.args.c_out, self.args.c_out, device=self.device, dtype=dtype), 
                            torch.eye(N, device=self.device, dtype=dtype))
        mask1 = mask1.unsqueeze(0).unsqueeze(0) - mask_base
        
        # mask2: 时空掩码（所有其他连接）
        mask2 = torch.ones((1, 1, L, L), device=self.device, dtype=dtype) - mask1 - mask0 - mask_base
        
        # 拼接三种掩码: [3, 1, L, L]
        masks = torch.cat([mask0, mask1, mask2], dim=0)
        return masks

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证函数
        在验证集上评估模型性能
        
        参数:
            vali_data: 验证数据集
            vali_loader: 验证数据加载器
            criterion: 损失函数
        
        返回:
            total_loss: 平均验证损失
        
        数据流和维度变化：
        1. 输入数据: batch_x [B, seq_len, C] - B:batch大小, C:通道数
        2. 模型输出: outputs [B, pred_len, C] - 预测未来pred_len个时间步
        3. 真实标签: batch_y [B, label_len+pred_len, C] -> 取后pred_len个
        """
        total_loss = []
        self.model.eval()  # 设置为评估模式
        
        with torch.no_grad():  # 不计算梯度
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # batch_x: [B, seq_len, C] - 输入序列
                batch_x = batch_x.float().to(self.device)
                # batch_y: [B, label_len+pred_len, C] - 包含标签和目标的完整序列
                batch_y = batch_y.float()

                # 模型前向传播
                # outputs: [B, pred_len, C] - 预测结果
                # _: moe_loss（验证时不使用）
                outputs, _ = self.model(batch_x, self.masks, is_training=False)
                
                # f_dim: 特征维度的起始位置
                # MS模式（多变量预测单变量）: -1表示只取最后一个特征
                # M/S模式（多变量预测多变量/单变量预测单变量）: 0表示取所有特征
                f_dim = -1 if self.args.features == 'MS' else 0
                
                # 只保留预测部分: [B, pred_len, C'] C'根据f_dim确定
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # 从batch_y中提取对应的真实值
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 移到CPU计算损失（节省GPU内存）
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # 计算损失
                loss = criterion(pred, true)
                total_loss.append(loss)
        
        # 计算平均损失
        total_loss = np.average(total_loss)
        self.model.train()  # 恢复训练模式
        return total_loss

    def train(self, setting):
        """
        训练函数
        执行完整的模型训练流程
        
        参数:
            setting: 实验设置字符串（用于保存路径）
        
        返回:
            self.model: 训练好的模型
        
        训练流程：
        1. 加载训练、验证和测试数据
        2. 初始化优化器、损失函数和早停机制
        3. 逐轮训练：
           - 前向传播获取预测
           - 计算损失（预测损失 + MoE辅助损失）
           - 反向传播更新参数
           - 验证和保存最佳模型
        """
        # 获取数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 初始化优化器和损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 如果使用自动混合精度训练
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 开始训练循环
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            # 遍历训练数据
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 数据维度说明：
                # batch_x: [B, seq_len, C] - 输入序列
                # batch_y: [B, label_len+pred_len, C] - 目标序列
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 模型前向传播
                # 维度变化详解：
                # 1. batch_x [B, seq_len, C] 输入模型
                # 2. 模型内部:
                #    - 归一化: [B, seq_len, C]
                #    - 转置重塑: [B, C, seq_len] -> [B, C*seq_len]
                #    - Patch嵌入: [B, L, D] 其中L=C*seq_len/patch_len
                #    - TimeFilter骨干: [B, L, D] (通过多层图过滤)
                #    - 预测头: [B, C, pred_len]
                #    - 转置: [B, pred_len, C]
                #    - 反归一化: [B, pred_len, C]
                # 3. outputs最终维度: [B, pred_len, C]
                outputs, moe_loss = self.model(batch_x, self.masks, is_training=True)

                # 提取预测部分和对应的真实值
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # 计算总损失 = 预测损失 + MoE辅助损失
                # MoE损失用于平衡专家的负载
                alpha = 0.05  # MoE损失的权重
                loss = criterion(outputs, batch_y) + alpha * moe_loss
                train_loss.append(loss.item())

                # 每100个batch打印一次训练状态
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播和参数更新
                if self.args.use_amp:
                    # 使用混合精度训练
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 标准反向传播
                    loss.backward()
                    model_optim.step()

            # 每轮结束后的统计和验证
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # 早停检查和模型保存
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        测试函数
        在测试集上评估模型并保存结果
        
        参数:
            setting: 实验设置字符串
            test: 是否从检查点加载模型 (0:使用当前模型, 1:加载保存的模型)
        
        测试流程和数据维度变化：
        1. 输入: batch_x [B, seq_len, C]
        2. 模型预测流程:
           a. 归一化输入
           b. 转置并重塑为patch: [B, C*seq_len] -> [B, L, D]
           c. 通过TimeFilter骨干网络处理
           d. 预测头映射: [B, C, L*D] -> [B, C, pred_len]
           e. 转置并反归一化: [B, pred_len, C]
        3. 输出: outputs [B, pred_len, C] - 未来pred_len步的预测值
        4. 评估: 与batch_y比较，计算MAE、MSE等指标
        """
        test_data, test_loader = self._get_data(flag='test')
        
        # 如果test=1，从检查点加载模型
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 存储预测结果
        preds = []  # 预测值列表
        trues = []  # 真实值列表
        inputs = []  # 输入值列表

        self.model.eval()  # 设置为评估模式
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 数据维度:
                # batch_x: [B, seq_len, C] - 输入序列
                # batch_y: [B, label_len+pred_len, C] - 目标序列
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 模型前向传播 - 详细的维度变化过程：
                # ===== 输入阶段 =====
                # batch_x: [B, seq_len, C] 例如 [32, 96, 7]
                
                # ===== 模型内部处理 =====
                # 1. 归一化: [B, seq_len, C] -> [B, seq_len, C]
                
                # 2. 转置和重塑:
                #    [B, seq_len, C] -> [B, C, seq_len] -> [B, C*seq_len]
                #    例如: [32, 96, 7] -> [32, 7, 96] -> [32, 672]
                
                # 3. Patch嵌入:
                #    unfold操作: [B, C*seq_len] -> [B, L, patch_len]
                #    其中 L = C*seq_len / patch_len
                #    例如: [32, 672] -> [32, 42, 16] (patch_len=16)
                #    
                #    线性投影: [B, L, patch_len] -> [B, L, D]
                #    例如: [32, 42, 16] -> [32, 42, 512] (D=d_model=512)
                #    
                #    添加位置编码: [B, L, D] -> [B, L, D]
                
                # 4. TimeFilter主干网络:
                #    通过多层GraphBlock处理: [B, L, D] -> [B, L, D]
                #    每层包含: 图学习 -> 图卷积 -> FFN
                #    例如: [32, 42, 512] -> [32, 42, 512]
                
                # 5. 预测头:
                #    重塑: [B, L, D] -> [B, C, num_patches, D]
                #    例如: [32, 42, 512] -> [32, 7, 6, 512] (num_patches=6)
                #    
                #    展平: [B, C, num_patches, D] -> [B, C, num_patches*D]
                #    例如: [32, 7, 6, 512] -> [32, 7, 3072]
                #    
                #    线性映射: [B, C, num_patches*D] -> [B, C, pred_len]
                #    例如: [32, 7, 3072] -> [32, 7, 96] (pred_len=96)
                #    
                #    转置: [B, C, pred_len] -> [B, pred_len, C]
                #    例如: [32, 7, 96] -> [32, 96, 7]
                
                # 6. 反归一化: [B, pred_len, C] -> [B, pred_len, C]
                
                # ===== 输出 =====
                # outputs: [B, pred_len, C] 例如 [32, 96, 7]
                # moe_loss: 标量，MoE的辅助损失
                outputs, _ = self.model(batch_x, self.masks, is_training=False)

                # 根据预测模式选择特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                
                # 只保留预测的部分
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                # 转换为numpy数组
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                
                # 如果需要，进行逆标准化（恢复到原始尺度）
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                # 根据特征类型提取相应的维度
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                # 保存结果
                input_ = batch_x
                pred = outputs
                true = batch_y

                inputs.append(input_)
                preds.append(pred)
                trues.append(true)

        # 拼接所有批次的结果
        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 重塑为标准格式
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # 创建结果保存目录
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 计算评估指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        # 保存结果到文件
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # 可选：保存详细的预测结果（已注释）
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'input.npy', inputs)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
