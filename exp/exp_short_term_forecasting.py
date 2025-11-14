"""
短期预测实验类
实现TimeFilter模型的短期时间序列预测任务（主要用于M4数据集）
与长期预测的区别：
1. 使用不同的损失函数（MAPE、MASE、SMAPE）
2. 针对M4数据集的特殊处理
3. 不同的验证和测试流程
"""

from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    """
    短期预测实验类
    
    专门用于M4短期预测竞赛数据集
    支持多种时间序列频率：小时、日、周、月、季度、年度
    
    参数:
        args: 命令行参数对象
    """
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """
        构建模型
        针对M4数据集自动配置序列长度和预测长度
        
        返回:
            model: TimeFilter模型实例
        """
        # M4数据集的特殊配置
        if self.args.data == 'm4':
            # 根据季节性模式设置预测长度
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            # 输入长度设为预测长度的2倍
            self.args.seq_len = 2 * self.args.pred_len
            # 标签长度等于预测长度
            self.args.label_len = self.args.pred_len
            # 设置频率映射
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        
        # 创建模型实例
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 多GPU支持
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        获取数据集和数据加载器
        
        参数:
            flag: 数据集类型 ('train', 'val', 'test')
        
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

    def _select_criterion(self, loss_name='MSE'):
        """
        选择损失函数
        
        参数:
            loss_name: 损失函数名称 ('MSE', 'MAPE', 'MASE', 'SMAPE')
        
        返回:
            criterion: 对应的损失函数
        """
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()  # 平均绝对百分比误差
        elif loss_name == 'MASE':
            return mase_loss()  # 平均绝对标度误差
        elif loss_name == 'SMAPE':
            return smape_loss()  # 对称平均绝对百分比误差

    def train(self, setting):
        """
        训练函数
        执行短期预测模型的训练流程
        
        参数:
            setting: 实验设置字符串
        
        返回:
            self.model: 训练好的模型
        
        数据维度说明：
        - batch_x: [B, seq_len, C] 输入序列
        - dec_inp: [B, label_len+pred_len, C] 解码器输入（前label_len是真实值，后pred_len是零）
        - outputs: [B, pred_len, C] 模型预测输出
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # 创建检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)  # 使用指定的损失函数
        mse = nn.MSELoss()  # 额外的MSE损失用于平滑性约束

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 数据维度:
                # batch_x: [B, seq_len, C] - 编码器输入
                # batch_y: [B, label_len+pred_len, C] - 目标序列
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构造解码器输入（用于某些模型架构）
                # dec_inp: [B, label_len+pred_len, C]
                # 前label_len部分使用真实值，后pred_len部分用零填充
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型前向传播
                # 注意：这里传入dec_inp而非masks（不同于长期预测）
                outputs = self.model(batch_x, None, dec_inp, None)

                # 提取预测部分
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # 计算损失
                # loss_value: 主要预测损失（MAPE/MASE/SMAPE）
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                # loss_sharpness: 平滑性损失（预测的一阶差分与真实值的一阶差分的MSE）
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), 
                                    (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                # 总损失（这里只使用主要损失，平滑性损失已注释）
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # 早停和模型保存
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        """
        验证函数
        在验证集上评估模型性能
        
        参数:
            train_loader: 训练数据加载器（用于获取最后的输入窗口）
            vali_loader: 验证数据加载器
            criterion: 损失函数
        
        返回:
            loss: 验证损失
        
        数据维度说明：
        - x: [B, seq_len, 1] 最后的输入窗口
        - dec_inp: [B, label_len+pred_len, 1] 解码器输入
        - outputs: [B, pred_len, 1] 预测输出
        """
        # 获取训练集最后的输入窗口作为验证的起点
        x, _ = train_loader.dataset.last_insample_window()
        # 获取验证集的真实时间序列
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # 添加特征维度: [B, seq_len] -> [B, seq_len, 1]

        self.model.eval()
        with torch.no_grad():
            # 构造解码器输入
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            
            # 初始化输出张量
            outputs = torch.zeros((B, self.args.pred_len, C)).float()
            
            # 分批次预测（避免内存溢出）
            id_list = np.arange(0, B, 500)  # 每次处理500个样本
            id_list = np.append(id_list, B)
            
            for i in range(len(id_list) - 1):
                # 对每个批次进行预测
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                ).detach().cpu()
            
            # 提取预测部分
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            # 计算损失
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, 
                           pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        """
        测试函数
        在测试集上评估模型并生成M4格式的预测结果
        
        参数:
            setting: 实验设置字符串
            test: 是否从检查点加载模型
        
        测试流程：
        1. 使用训练集最后的窗口作为输入
        2. 生成测试集的预测
        3. 保存预测结果为CSV格式
        4. 如果所有6个M4任务完成，计算综合指标
        
        数据维度说明：
        - x: [B, seq_len, 1] 输入序列
        - dec_inp: [B, label_len+pred_len, 1] 解码器输入
        - outputs: [B, pred_len, 1] 预测输出
        """
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        
        # 获取输入数据
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        # 加载模型
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 创建保存目录
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # 构造解码器输入
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            
            # 初始化输出
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            
            # 逐个样本预测（为了稳定性）
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            
            for i in range(len(id_list) - 1):
                # 预测单个样本
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]], 
                    None
                )

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            # 提取预测结果
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            # 可视化部分预测结果（每10%采样一个）
            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                # 拼接输入和预测结果
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # 保存M4格式的预测结果
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 创建DataFrame并保存为CSV
        forecasts_df = pandas.DataFrame(preds[:, :, 0], 
                                       columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        
        # 检查是否所有6个M4任务都已完成
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            # 计算M4竞赛的综合评估指标
            m4_summary = M4Summary(file_path, self.args.root_path)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)  # 对称平均绝对百分比误差
            print('mape:', mape)  # 平均绝对百分比误差
            print('mase:', mase)  # 平均绝对标度误差
            print('owa:', owa_results)  # 总体加权平均
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
