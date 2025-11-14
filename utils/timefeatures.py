"""
时间特征提取模块
从时间戳中提取各种时间特征（小时、天、月等）用于模型输入

本模块来源于GluonTS项目，并在Apache License 2.0许可下使用
From: gluonts/src/gluonts/time_feature/_base.py
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
"""

# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List  # 类型提示

import numpy as np  # 数值计算
import pandas as pd  # 数据处理
from pandas.tseries import offsets  # 时间偏移
from pandas.tseries.frequencies import to_offset  # 频率转换


class TimeFeature:
    """
    时间特征基类
    定义时间特征提取的接口
    """
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        从时间索引中提取特征
        
        参数:
            index: pandas时间索引
        返回:
            提取的特征数组（归一化到[-0.5, 0.5]）
        """
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    分钟中的秒数特征
    将秒数编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    小时中的分钟数特征
    将分钟数编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    一天中的小时数特征
    将小时数编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    一周中的天数特征
    将星期几编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    一月中的天数特征
    将日期编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    一年中的天数特征
    将一年中的第几天编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    一年中的月份特征
    将月份编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    一年中的周数特征
    将周数编码为[-0.5, 0.5]之间的值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    根据频率字符串返回相应的时间特征列表
    
    参数:
        freq_str: 频率字符串，格式为[倍数][粒度]，如"12H"、"5min"、"1D"等
    
    返回:
        时间特征对象的列表
    
    支持的频率：
        - Y/A: 年度
        - M: 月度
        - W: 周度
        - D: 日度
        - B: 工作日
        - H: 小时
        - T/min: 分钟
        - S: 秒
    """
    # 根据不同的时间偏移类型，定义需要提取的时间特征
    features_by_offsets = {
        offsets.YearEnd: [],  # 年度数据不需要额外特征
        offsets.QuarterEnd: [MonthOfYear],  # 季度数据需要月份特征
        offsets.MonthEnd: [MonthOfYear],  # 月度数据需要月份特征
        offsets.Week: [DayOfMonth, WeekOfYear],  # 周度数据需要日期和周数特征
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # 日度数据需要星期、日期、年中天数
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # 工作日同日度
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # 小时级数据需要小时和日期特征
        offsets.Minute: [  # 分钟级数据需要分钟、小时和日期特征
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [  # 秒级数据需要秒、分钟、小时和日期特征
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # 将频率字符串转换为pandas的offset对象
    offset = to_offset(freq_str)

    # 根据offset类型返回相应的特征列表
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # 如果频率不支持，抛出错误并提示支持的频率
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    从日期中提取时间特征
    
    参数:
        dates: 日期序列
        freq: 时间频率（默认为小时'h'）
    
    返回:
        时间特征矩阵，每一行对应一个时间特征
    """
    # 获取对应频率的特征提取器列表
    # 对每个特征提取器应用到dates上，然后垂直堆叠结果
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
