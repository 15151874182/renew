# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:10:34 2022

@author: zhangqipei
"""

"""
调用center/feature文件夹下的特征工程方法，
进行特征工程方案的编写
"""

import pandas as pd
from center.feature.feature_engineer import add_feat_sin
from center.tools.logger_function import get_logger
from zj_jiaxing.config import config_parser, global_config
import sklearn.preprocessing as preprocessing
import numpy as np
import os
logger = get_logger()

def generate_month_feature_series(series):
    series = pd.to_datetime(series)
    months = series.apply(lambda x: x.month)
    return months

def generate_month_feature(series):
    """
    输入时间列
    """
    months = generate_month_feature_series(series)

    # 处理成one hot特征
    months_feature = pd.get_dummies(months, prefix="months")
    months_feature.index = series.index
    months_feature["months"] = months

    for i in range(12):
        name = f"months_{i}"
        if name not in months_feature.columns:
            months_feature[name] = 0
    return months_feature


def generate_date_feature(df, time_name="date"):
    df[time_name] = pd.to_datetime(df[time_name])
    # 时间部分
    df['time_part'] = df[time_name].apply(lambda x: x.strftime('%H:%M:%S'))

    # 月份 1-12
    df['month'] = df[time_name].apply(lambda x: x.month)
    # 季节
    df.loc[df['month'].apply(lambda x: (x >= 3) & (x <= 5)), 'season'] = 'spring'
    df.loc[df['month'].apply(lambda x: (x >= 6) & (x <= 8)), 'season'] = 'summer'
    df.loc[df['month'].apply(lambda x: (x >= 9) & (x <= 11)), 'season'] = 'autumn'
    df['season'] = df['season'].fillna('winter')

    # 日期部分
    df['date_part'] = df[time_name].apply(lambda x: x.strftime('%Y-%m-%d'))

    # 时间部分
    df['time_part'] = df[time_name].apply(lambda x: x.strftime('%H:%M:%S'))

    # 小时
    df['hour'] = df[time_name].apply(lambda x: x.hour)

    # 一天中的分钟数
    df['minute'] = df[time_name].apply(lambda x: x.minute).astype(int) + df['hour'].astype(int) * 60

    # 时段
    df.loc[df['time_part'].apply(lambda x: (x >= '06:00:00') & (x < '10:00:00')), 'section'] = '1'
    df.loc[df['time_part'].apply(lambda x: (x >= '10:00:00') & (x < '14:00:00')), 'section'] = '2'
    df.loc[df['time_part'].apply(lambda x: (x >= '14:00:00') & (x < '18:00:00')), 'section'] = '3'
    df['section'] = df['section'].fillna('0')

    # 字符类型数据进行编码
    le = preprocessing.LabelEncoder()

    le.fit(df['section'])
    df['section_encode'] = le.transform(df['section']).astype(np.float)

    le.fit(df['season'])
    df['season_encode'] = le.transform(df['season']).astype(np.float)

    return df

def add_wd_feature(config, df_feat):
    """
    添加风向sin、cos特征

    Parameters
    ----------
    config: object
        特定场站的配置文件对象
    df_feat : dataframe
        场站特征数据集.

    Returns
    -------
    df_feat: pandas.DataFrame
        融合风向后的场站特征数据集

    """
    # 根据场站名称来获取对应的配置文件
    # config = config_parser(station_id)

    # 获取场站对应配置文件中的风向列名组成的list
    wind_direction_feas = config.get_para("wind_direction_feas")

    # 防止数据中出现没有的风向特征列
    df_feat_cols = df_feat.columns.to_list()
    # 取两个col list的交集
    wd_unite = list(set(df_feat_cols) & set(wind_direction_feas))

    if len(wd_unite) > 0:
        # 添加sin、cos特征
        df_feat = add_feat_sin(df_feat, wd_unite)

    return df_feat

def diff_weather(df,
                 cols = ["ws10", "ws30", "ws31", "ws32", "swr", "lwr"]
                 ):
    """ 根据论文《考虑风电并网的超短期负荷预测》P8，风速变化也是一个重要的统计特征"""
    for c in cols:
        new_col = f"{c}_diff"
        df[new_col] = df[c].diff(periods=1)
        df[new_col].fillna(method="bfill", inplace=True)

        new_col_ = f"{c}_diff_reverse"
        df[new_col_] = df[c].diff(periods=-1)
        df[new_col_].fillna(method="ffill", inplace=True)
    return df


def feat_combine(config, df_feat):
    """
    添加自定义融合特征

    Parameters
    ----------
    config: object
        特定场站的配置文件对象
    df_feat: pandas.DataFrame
        场站特征数据集

    Returns
    -------
    df_feat: pandas.DataFrame
        场站融合特征数据集
    """

    # 根据场站名称来获取对应的配置文件
    # config = config_parser(station_id)
    # 获取场站对应配置文件中的风向列名组成的list
    wind_direction_feas = config.get_para("wind_direction_feas")

    logger.debug('enter feat_combine_manu')
    # feature_cols = df.columns
    # time_name = config.time_name

    # 添加风向的sin、cos特征
    df_feat = add_feat_sin(df_feat, wind_direction_feas)

    # diff 特征
    diff_col = [c for c in df_feat.columns if "speed" in c]
    df_feat = diff_weather(df_feat, diff_col)

    df_feat["months"] = generate_month_feature_series(df_feat["date"])
    df_feat = generate_date_feature(df_feat, "date")

    # TODO:这部分函数就不在center里，因为不具备通用性，目前来看效果一般，等后期再恢复
    # features = df.columns
    # df = add_feat_add_and_multi_divide(df,features)

    # # 添加平方项特征
    # df = add_squ_fea(df, config.wind_power_names)

    # # 添加时间类特征
    # df = add_time_fea(df, df_time)

    # # 添加增量信息
    # df = add_diff_fea(df, config.clean_feature_cols)

    # # 添加3次方信息
    # df = add_cube_fea(df, config.wind_power_names)

    # logger.debug('predict_feature_select: ', df.columns)

    return df_feat

# train用实测，predict用预测天气
# def data_generator(config, train=True, weather_type="nwp", split=True):
#     """
#     根据场站名称，生成训练或预测使用的数据格式
#
#     Parameters
#     ----------
#     config: object
#         特定场站的配置文件对象
#     train : bool, optional. The default is True.
#         True-生成训练格式数据
#         False-生成预测格式数据
#     weather_type : str, optional
#         使用的天气数据类型，nwp or real.
#
#     Returns
#     -------
#     tuple:
#         df : pandas.DataFrame
#             特征列组成的df，实际等于df_feature.
#         df_label : pandas.DataFrame
#             标签组成的df.
#         df_time : pandas.DataFrame
#             时间列组成的df.
#     """
#
#     # \u6839\u636e\u573a\u7ad9\u540d\u79f0\u6765\u83b7\u53d6\u5bf9\u5e94\u7684\u914d\u7f6e\u6587\u4ef6
#     # config = config_parser(station_id)
#     # \u83b7\u53d6\u914d\u7f6e\u6587\u4ef6\u4e2d\u7684\u4fe1\u606f
#     time_name = config.get_para("time_name")
#     label_col_name = config.get_para("label_col_name")
#
#     # \u83b7\u53d6\u5bf9\u5e94\u573a\u7ad9\u7684\u529f\u7387\u548c\u5929\u6c14\u6570\u636e
#     data_path = os.path.join(config.station_path, "data")
#     load_path = os.path.join(data_path, "history_cleaned_real_load.csv")
#     if train:
#         weather_path = os.path.join(data_path, "history_cleaned_real_weather.csv")
#     else:
#
#         if len(weather_type):
#             weather_path = os.path.join(data_path, f"history_cleaned_XXL_{weather_type}_weather.csv")
#         else:
#             # \u6bcf\u5929\u4f20\u4ec0\u4e48\u9884\u6d4b\u4ec0\u4e48
#             weather_path = os.path.join(data_path, f"daily_cleaned_pred_weather.csv")
#         # weather_path = os.path.join(data_path, f"history_cleaned_XXL_{weather_type}_weather.csv")
#
#     # load_path = f'{config.get_para("station_path")}/data/history_cleaned_real_load.csv'
#     # weather_path = f'{config.get_para("station_path")}/data/history_cleaned_XXL_{weather_type}_weather.csv'
#
#     if train:
#         # \u8fdb\u5165\u5230\u8bad\u7ec3\u96c6\u751f\u6210\u7684\u6d41\u7a0b,\u9700\u8981\u5b9e\u6d4bload\u548cweather
#         load = pd.read_csv(load_path)
#         load = load.dropna(how="any", axis=0)
#         load[time_name] = pd.to_datetime(load[time_name])
#
#         weather = pd.read_csv(weather_path)
#         weather = weather.dropna(how="any", axis=0)
#         weather[time_name] = pd.to_datetime(weather[time_name])
#
#         # merge\u529f\u7387\u548c\u5929\u6c14\uff0c\u5e76\u4f9d\u636e\u914d\u7f6e\u6587\u4ef6\u9009\u53d6\u76f8\u5e94\u5217
#         df = pd.merge(left=load, right=weather, on=time_col_name,
#                       how="inner").reset_index(drop=True)
#
#         # \u7279\u5f81\u878d\u5408\u5224\u65ad\uff0c\u5982\u679c\u4e0d\u505a\u7279\u5f81\u878d\u5408\uff0c\u5c31\u7b80\u5355\u5bf9\u98ce\u5411sin\u3001cos\u5904\u7406
#         if config.get_para("feat_combine"):
#             df = feat_combine(config, df)
#         else:
#             # \u5982\u679c\u4e0d\u7528\u7279\u5f81\u878d\u5408\uff0c\u5c31\u7b80\u5355\u7684\u5bf9\u98ce\u5411\u8fdb\u884csin\u3001cos\u5904\u7406
#             df = add_wd_feature(config, df)
#
#         if split:
#             # \u5212\u5206label\u3001feature\u3001time\u5217
#             df_time = df.pop(time_col_name)
#             df_label = df.pop(label_col_name)
#
#             return df, df_label, df_time
#         return df
#
#     else:
#         # \u8fdb\u5165\u5230\u9884\u6d4b\u6570\u636e\u96c6\u751f\u6210\u6d41\u7a0b\uff0c\u53ea\u9700\u8981weather
#         df = pd.read_csv(weather_path)
#         df = df.dropna(how="any", axis=0)
#         df[time_col_name] = pd.to_datetime(df[time_col_name])
#
#         # \u7279\u5f81\u878d\u5408\u5224\u65ad\uff0c\u5982\u679c\u4e0d\u505a\u7279\u5f81\u878d\u5408\uff0c\u5c31\u7b80\u5355\u5bf9\u98ce\u5411sin\u3001cos\u5904\u7406
#         if config.get_para("feat_combine"):
#             df = feat_combine(config, df)
#         else:
#             # \u5982\u679c\u4e0d\u7528\u7279\u5f81\u878d\u5408\uff0c\u5c31\u7b80\u5355\u7684\u5bf9\u98ce\u5411\u8fdb\u884csin\u3001cos\u5904\u7406
#             df = add_wd_feature(config, df)
#
#         return df

# 训练拼接实测和预测天气的版本
def data_generator(config, train=True, weather_type="nwp", split=True):
    """
    根据场站名称，生成训练或预测使用的数据格式

    Parameters
    ----------
    config: object
        特定场站的配置文件对象
    train : bool, optional. The default is True.
        True-生成训练格式数据
        False-生成预测格式数据
    weather_type : str, optional
        使用的天气数据类型，nwp or real.

    Returns
    -------
    tuple:
        df : pandas.DataFrame
            特征列组成的df，实际等于df_feature.
        df_label : pandas.DataFrame
            标签组成的df.
        df_time : pandas.DataFrame
            时间列组成的df.
    """

    # 根据场站名称来获取对应的配置文件
    # config = config_parser(station_id)
    # 获取配置文件中的信息
    time_name = config.get_para("time_name")
    load_name = config.get_para("load_name")

    # 获取对应场站的功率和天气数据
    data_path = os.path.join(config.station_path, "data")
    
    real_load=os.path.join(data_path,'history_cleaned_real_load.csv')
    nwp_weather=os.path.join(data_path,'history_cleaned_nwp_weather.csv')
    daily_weather=os.path.join(data_path,'daily_weather_from_db.csv')
    if train:
        # 进入到训练集生成的流程,需要实测load和weather
        load = pd.read_csv(real_load)
        load[time_name] = pd.to_datetime(load[time_name])
        weather = pd.read_csv(nwp_weather)          
        weather[time_name] = pd.to_datetime(weather[time_name])

        # merge功率和天气，并依据配置文件选取相应列
        df = pd.merge(left=load, right=weather, on=time_name,
                      how="left").reset_index(drop=True)
        df=df.dropna().reset_index(drop=True)
        # 特征融合判断，如果不做特征融合，就简单对风向sin、cos处理
        if config.get_para("feat_combine"):
            df = feat_combine(config, df)
        else:
            # 如果不用特征融合，就简单的对风向进行sin、cos处理
            df = add_wd_feature(config, df)

        if split:
            # 划分label、feature、time列
            df_time = df.pop(time_name)
            df_label = df.pop(load_name)

            return df, df_label, df_time
        return df

    else:
        # 进入到预测数据集生成流程，只需要weather
        df = pd.read_csv(daily_weather)
        df[time_name] = pd.to_datetime(df[time_name])

        # 特征融合判断，如果不做特征融合，就简单对风向sin、cos处理
        if config.get_para("feat_combine"):
            df = feat_combine(config, df)
        else:
            # 如果不用特征融合，就简单的对风向进行sin、cos处理
            df = add_wd_feature(config, df)

        return df
