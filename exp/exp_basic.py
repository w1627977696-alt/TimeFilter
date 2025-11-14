"""
实验基类模块
定义了所有实验类的基本结构和接口
"""

import os  # 操作系统接口
import torch  # PyTorch深度学习框架
from models import TimeFilter  # TimeFilter模型


class Exp_Basic(object):
    """
    实验基类
    所有具体实验类（如长期预测、短期预测等）的父类
    定义了实验的基本流程和接口
    
    参数:
        args: 命令行参数对象，包含所有配置信息
    """
    def __init__(self, args):
        self.args = args  # 保存参数
        # 模型字典：将模型名称映射到模型模块
        self.model_dict = {
            'TimeFilter': TimeFilter
        }
        # 获取计算设备（GPU或CPU）
        self.device = self._acquire_device()
        # 构建模型并移动到指定设备
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        构建模型（抽象方法，由子类实现）
        
        返回:
            构建好的模型实例
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        获取计算设备
        根据配置决定使用GPU还是CPU
        
        返回:
            torch.device对象
        """
        if self.args.use_gpu:
            # 设置可见的CUDA设备
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # 创建GPU设备对象
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # 使用CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        """
        获取数据（抽象方法，由子类实现）
        """
        pass

    def vali(self):
        """
        验证方法（抽象方法，由子类实现）
        """
        pass

    def train(self):
        """
        训练方法（抽象方法，由子类实现）
        """
        pass

    def test(self):
        """
        测试方法（抽象方法，由子类实现）
        """
        pass
