# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:06:46 2022

@author: cty、zqp
"""
import numpy as np
import os
import pandas as pd
from scipy import interpolate
from center.dataclean.newEnergy_DataClean import Clean
from center.tools.logger_function import get_logger
import datetime
from scipy.signal import argrelextrema
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
logger = get_logger()

def fix_date(df_cleaned,df,freq='15T'):
    '''
    df必须包含df['date']列，格式最好为datetime标准格式！原始数据输入的时间可能不连贯，修补时间，若已经连贯则无影响

    Parameters
    ----------
    df : dataFrame
        DESCRIPTION.
    ----------
    freq : int
        时间间隔.
    Returns
    -------
    df : dataFrame
        DESCRIPTION.

    '''
    time=pd.DataFrame(pd.date_range(start=df['date'].iloc[0], end=df['date'].iloc[-1],freq=freq))
    time.columns=['date']
    df_cleaned=pd.merge(df_cleaned,time,on='date',how='right')        
    return df_cleaned

def plot_peroid(df,filename,time_col = "datetime",cols = ["actual","predict_load"],start_day = "2021-11-12",end_day=None,days = 30):
    """
    按指定日期进行画图
    start_day: 设置起始的日期
    days:从起始日开始，一共要画多少天
    """
    #将cols列全部转换成数值类型
    for col in cols:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
    
    #选择我们想要的列
    add_time = copy.deepcopy(cols)
    add_time.append(time_col)
    df = df.loc[:,add_time]
    
    #时间列转化成时间格式
    df[time_col] = pd.to_datetime(df[time_col])
    
    start_day = pd.to_datetime(start_day)
    
    #画图进行对比,设置画布大小
    plt.figure(figsize=(30,10))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    #取需要画图的时间段数据
    if end_day ==None:
        end_day = start_day + pd.Timedelta(days=days)
    else:
        end_day = pd.to_datetime(end_day)
    temp = df[np.logical_and((df[time_col]>=start_day),(df[time_col]<end_day))]
    #画图
    x = temp.loc[:,time_col]
    #画出所有的列
    for col in cols:
        plt.plot(x, temp.loc[:,col],label=col,alpha=0.6)
    print(f"画图：{filename}")
    plt.legend(loc="upper left",fontsize='x-large')
    plt.title(f"{filename}",fontsize='x-large')
    plt.savefig(f"./figure/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
    
def concat_base(config,new_df,data_type,data_name="weather"):
    """
    注意：new_df必须是清洗后时间戳保证正确的数据
    将清洗后的new_df，增量追加到base_df中并保存。

    Parameters
    ----------
    config : object
        该station_id的配置实例.
    new_df : DataFrame
        新增的需要追加到base的数据.
    data_type : str
        实测or预测   real or nwp or fore
    data_name : str
        weather or load
    gys_name : str
        XXL or XBY

    """
    #根据配置文件获取场站的文件夹路径
    station_path = config.get_para("station_path")
    date_col = config.get_para("time_name")
    
    if data_name == "weather":
        if data_type == "nwp":
            base_path = f"{station_path}/data/history_cleaned_{data_type}_{data_name}.csv"
        elif data_type == "real":
            base_path = f"{station_path}/data/history_cleaned_{data_type}_{data_name}.csv"
    
    elif data_name == "load":
        if data_type == "fore":
            base_path = f"{station_path}/data/history_cleaned_{data_type}_{data_name}.csv"
        elif data_type == "real":
            base_path = f"{station_path}/data/history_cleaned_{data_type}_{data_name}.csv"

    if not os.path.exists(base_path):
        new_df.to_csv(base_path, index=False)
        return
    df_base = pd.read_csv(base_path)
    
    #保证列名称顺序和原df_base一致
    new_df = new_df.loc[:,df_base.columns]
    df_base = pd.concat([df_base,new_df],axis=0).reset_index(drop=True)
    
    #去除时间戳可能存在的重复
    df_base[date_col] = pd.to_datetime(df_base[date_col])
    # 删除时间戳重复数据
    df_base.drop_duplicates(subset=[date_col], keep='last', inplace=True)
    # 对数据按时间戳进行排序
    df_base.sort_values(by=date_col, kind='mergsort', ascending=True, inplace=True)
    # 排序后的index重置
    df_base.reset_index(drop=True, inplace=True)
    
    df_base.to_csv(base_path,index=False)
    
    logger.debug(f"concat new data into :{base_path}")


def get_cleaner(config, use_type):
    '''
    创建数据清洗方法的对象

    Parameters
    ----------
    config : object
        该station_id的配置实例.
    use_type : str
        'train' or 'test'.

    Returns
    -------
    cleaner : object
        清洗类的实例.

    '''
    
    cleaner=Clean(station_type=config.get_para("station_type"),
                  use_type=use_type,
                  station_name=config.area_name,
                  capacity=config.get_para("capacity"),
                  freq=config.get_para("day_point"),
                  longitude=config.get_para("longitude"),
                  latitude=config.get_para("latitude"),
                  similarity_detect= True if use_type=='train' else False,
                  threshold=config.get_para("threshold")
                    )
    return cleaner

def clean_station(config, use_type, df, set_cols=[]):
    '''
    针对单场站的数据清洗

    Parameters
    ----------
    config : object
        该station_id的配置实例.
    use_type : str
        'train' or 'test'.
    df : dataframe
        要清洗的数据.

    Returns
    -------
    df_cleaned : dataframe
        清洗完的数据.
    del_info : dataframe
        被清洗数据的信息.

    '''
    cleaner=Clean(station_type=config.get_para("station_type"),
                  use_type=use_type,
                  station_name=config.area_name,
                  capacity=config.get_para("capacity"),
                  freq=config.get_para("day_point"),
                  longitude=config.get_para("longitude"),
                  latitude=config.get_para("latitude"),
                  similarity_detect= True if use_type=='train' else False,
                  threshold=config.get_para("threshold")
                    )
    df_cleaned,del_info = cleaner.clean_station(
                                df,
                                clean_col_list=config.get_para("clean_feature_cols") if not set_cols else set_cols,
                                time_col='date',
                                load_col='load' if use_type=='train' else None)
    return df_cleaned,del_info

def solve_pv_frog_leg(df,points=4):
    '''
    Parameters
    ----------
    df : dataframe
        数据.
    points : int, optional
        从起始点后连续几个点使用spline插值. The default is 4.

    Returns
    -------
    df : dataframe

    '''
    for i in range(2, len(df['load'])):
        if df['load'][i] !=0 and df['load'][i-1] ==0 and df['load'][i-2] ==0: ##找到日出的起点
            df['load'][i:i+points]=np.nan
        if df['load'][i] !=0 and df['load'][i+1] ==0 and df['load'][i+2] ==0: ##找到日落的起点
            df['load'][i-points:i]=np.nan     
    df["load"]=df["load"].interpolate(method='spline', order=3) ##spline插值法
    return df



def IMF(load, length=96, col='actual'):
    """
    功能：
        寻找输入时间序列的IMF
    输入：
        load 输入的负荷时间序列,['datetime'col]
        length = 96  设置区间长度,用于寻找区间极值
    返回：
        包含IMF的一张总表，Dataframe
    """

    load.loc[:, 't'] = load.index

    # 找到每个区间内中的极大值、极小值及对应的时刻t
    max_index = []
    min_index = []

    # 找到区间极值的索引值
    for day in range(len(load) // length):
        temp = load.iloc[day * length:(day + 1) * length, :]
        max_index.append(temp[col].idxmax(axis=0))
        min_index.append(temp[col].idxmin(axis=0))

    max_value = load.loc[max_index, [col, "t"]]
    min_value = load.loc[min_index, [col, "t"]]
    # 处理极大值插值
    # 定义一个三次样条插值方法的函数
    cb = interpolate.interp1d(max_value['t'].values,
                              max_value[col].values, kind='cubic')
    # 定义插值区间t
    t_range = np.arange(max_value['t'].min(), max_value['t'].max(), 1)

    # 区间范围外的插值我们设置为实际值,并将插值的值写入到总表中
    load.loc[:, 'interp_max'] = load[col]
    load.loc[t_range, 'interp_max'] = cb(t_range)

    # 处理极小值插值
    # 定义一个三次样条插值方法的函数
    cb = interpolate.interp1d(min_value['t'].values,
                              min_value[col].values, kind='cubic')
    # 定义插值区间t
    t_range = np.arange(min_value['t'].min(), min_value['t'].max(), 1)
    # 区间范围外的插值我们设置为实际值,并将插值的值写入到总表中
    load.loc[:, 'interp_min'] = load[col]
    load.loc[t_range, 'interp_min'] = cb(t_range)

    # 求出上下包络线的平均值m(t)，在原时间序列中减去它:h(t)=x(t)-m(t)
    load.loc[:, 'mean'] = (load['interp_max'].values +
                           load['interp_min'].values) / 2
    load.loc[:, 'h(t)'] = (load[col].values - load['mean'].values)

    return load



def smooth_pv(df,day_len):
    """
    强制平滑光伏出力数据

    Parameters
    ----------
    df : DataFrame
        待处理的数据结果.
    day_len : int
        一天数据的长度.
    """
    def smooth_oneday(df):
        """
        要求数据必须是非零时段，只处理一天的数据
        """
            
        #找到第一个不是0
        for i in range(len(df)):
            if df["load"].iloc[i] != 0:
                start = i
                break
        #找到第最后一个不是0
        for i in range(len(df)):
            if df["load"].iloc[i] != 0 and df["load"].iloc[i+1] == 0:
                end = i
        
        temp = df.iloc[start:end,:]
        temp = IMF(temp, length=12, col='load')
    
        df["load"].iloc[start:end] = temp["mean"].values
        
        return df
    
    day_len = 96
    for i in range(len(df)//day_len):
        temp = df.iloc[96*i:96*(i+1),:]
        temp = smooth_oneday(temp)
        # 替换平滑后的值
        df["load"].iloc[96*i:96*(i+1)] = temp["load"].values
        
    return df

####clean demo
if __name__ == '__main__':
    from dataclean import clean_station
    from config import config_parser
    f='./test.txt'
    df=pd.read_table(f)
    df=df[['Time', 'rtPower(MW)', 'speed10', 'dir10', 'DirectRadiation','tempXXL',
    'humXXL', 'pressXXL']]
    df.rename(columns={'Time':'date','rtPower(MW)':'load','DirectRadiation':'rad'},inplace=True)
    config = config_parser('TKYTGF')
    df_cleaned, del_info = clean_station(config, use_type="train", df=df, set_cols=[])





