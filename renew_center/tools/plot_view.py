# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 08:55:05 2020

@author: zhangqipei
"""

"""
画图看下湛江地区220KV母线负荷的形状
"""
import os
import copy
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy import stats
import matplotlib as mpl
# mpl.use("AGG")
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
# from sklearn.preprocessing import MinMaxScaler



"""
画同一地区年、月、周的曲线，观察是否存在规律性
"""

def plot_years(df,filename,start_day = "2018-1-1",years = 3):
    """
    按年画图，观察年规律
    start_day: 设置起始的日期
    years: 设置取连续几周的数据进行画图对比
    """
    #给曲线命名
    year_list = ["2018","2019","2020"]   
    #重命名列名
    df.columns = ["date","load"]
    
    #时间列转化成时间格式
    df["date"] = pd.to_datetime(df["date"])
    
    start_day = pd.to_datetime(start_day)
    
    #画图进行对比,设置画布大小
    plt.figure(figsize=(18,9))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    for i in range(years):
        print("正在处理{}/{} year".format(i+1,years))
        #每次取一周的数据进行画图    
        end_day = start_day + relativedelta(years=1)
        temp = df[np.logical_and((df["date"]>=start_day),(df["date"]<end_day))]
        #画图
        x = range(len(temp))
        plt.plot(x, temp.loc[:,"load"],label="load_{}".format(year_list[i]),alpha=0.6)
        
        #更新start_day
        start_day = start_day + relativedelta(years=1)
        
    plt.legend(loc="upper left")
    plt.title("对比")
    plt.savefig("./figures/{}/年趋势图/{}.png".format(filename,filename))   
    plt.close()
    # plt.show()

def plot_weeks(df,filename,col="col",start_day = "2020-10-12", weeks = 2):
    """
    按周画图，观察周规律
    start_day: 设置起始的日期
    weeks: 设置取连续几周的数据进行画图对比
    """  
    #选择我们想要的列
    cols = ["date",col]
    df = df.loc[:,cols]
    
    #时间列转化成时间格式
    df["date"] = pd.to_datetime(df["date"])
    
    start_day_ori = pd.to_datetime(start_day)
    start_day = pd.to_datetime(start_day)

    #画图进行对比,设置画布大小
    plt.figure(figsize=(18,9))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    for i in range(weeks):
        print("正在处理{}/{} week".format(i+1,weeks))
        #每次取一周的数据进行画图
        end_day = start_day + timedelta(days=7)
        temp = df[np.logical_and((df["date"]>=start_day),(df["date"]<end_day))]
        #画图
        x = range(len(temp))
        plt.plot(x, temp.loc[:,col],label="{}_week{}".format(col,i+1))
        
        #更新start_day
        start_day = start_day + timedelta(days=7)
        
    plt.legend(loc="upper left")
    plt.title("{}起周趋势对比".format(start_day_ori.strftime("%Y-%m-%d")))
    # plt.savefig("./周趋势图/{}.png".format(filename[:-4]))    
    # plt.savefig("./figures/{}/周趋势图/{}.png".format(filename,filename))
    # plt.close()
    plt.show()


def plot_sync(df,filename,start_day = "2018-1-1",years = 3):
    """
    按天采样同一时刻的点画图，观察规律
    start_day: 设置起始的日期
    weeks: 设置取连续几周的数据进行画图对比
    """  
    #给曲线命名
    year_list = ["2018","2019","2020"]   
    #重命名列名
    df.columns = ["date","load"]
    
    #时间列转化成时间格式
    df["date"] = pd.to_datetime(df["date"])
    

    start_day = "2018-1-1"
    start_day = pd.to_datetime(start_day)
    
    #画图进行对比,设置画布大小
    plt.figure(figsize=(18,9))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    for i in range(years):
        print("正在处理{}/{} year".format(i+1,years))
        #每次取一周的数据进行画图    
        end_day = start_day + relativedelta(years=1)
        temp = df[np.logical_and((df["date"]>=start_day),(df["date"]<end_day))]
        #选择一天0-96中画哪个点进行对比
        temp = temp.iloc[48::96]
        x = range(len(temp))
        plt.plot(x, temp.loc[:,"load"],label="load_{}".format(year_list[i]),alpha=0.6)
        
        #更新start_day
        start_day = start_day + relativedelta(years=1)
        
    plt.legend(loc="upper left")
    plt.title("对比")
    plt.savefig("./figures/{}/按天采样趋势图/{}.png".format(filename,filename))
    # plt.savefig("./按天采样趋势图/{}.png".format(filename[:-4]))    
    plt.close()



def plot_peroid(df,filename,time_col = "datetime",cols = ["actual","predict_load"],start_day = "2021-11-12",end_day=None,days = 30):
    """
    按指定日期进行画图
    start_day: 设置起始的日期
    days:从起始日开始，一共要画多少天
    """
    #将cols列全部转换成数值类型
    df['date']=df.index
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
        plt.plot(x, temp.loc[:,col],label=col,alpha=1,linewidth =1.5)
    print(f"画图：{filename}")
    plt.legend(loc="upper left",fontsize='x-large')
    plt.title(f"{filename}",fontsize='x-large')
    # plt.savefig(f"./figure/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()
    plt.close()




def plot_series(df,filename,start_day = "2020-1-1",period = 30):
    """
    按指定的周期，进行连续的画图，用于观察数据的趋势和明显异常值
    start_day: 设置起始的日期
    period:默认的周期为30天
    """

    #重命名列名
    df.columns = ["date","load"]
    
    #时间列转化成时间格式
    df["date"] = pd.to_datetime(df["date"])
    
    start_day = pd.to_datetime(start_day)
    
    df_date = df["date"].to_frame() 
    while True:
        #判断终止条件
        if start_day > df_date.iloc[-1,0]:
            break;
        print("processing: {}".format(start_day))
        #画图进行对比,设置画布大小
        plt.figure(figsize=(18,9))
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False
        
        #取需要画图的时间段数据   
        end_day = start_day + relativedelta(days=period)
        temp = df[np.logical_and((df["date"]>=start_day),(df["date"]<end_day))]
        #画图
        x = temp.loc[:,"date"]
        plt.plot(x, temp.loc[:,"load"],label="load",alpha=0.6)
        
            
        plt.legend(loc="upper left")
        plt.title("对比{}-{}".format(start_day,end_day))
        # plt.savefig("./连续周期图/{}_{}.png".format(filename[:-4],start_day.strftime('%Y-%m-%d)')))   
        plt.savefig("./figures/{}/连续周期图/{}.png".format(filename,start_day.strftime('%Y-%m-%d)')))
        plt.close()
        # plt.show()
        
        #更新start_day
        start_day = end_day


def plot_ratio(df,filename,start_day = "2020-1-1",days =300 ):
    """
    功能：
        计算数据中相邻点的环比，并画图
    filename:
        str,被处理文件的文件名
    start_day：
        datetime,画图的数据起始日期
    days:
        int,从起始日期开始选择的连续天数
    """
    
    #重命名列名
    df.columns = ["date","load"]
    
    #时间列转化成时间格式
    df.loc[:,"date"] = pd.to_datetime(df.loc[:,"date"])
    
    start_day = pd.to_datetime(start_day)
    
    #取需要画图的时间段数据   
    end_day = start_day + relativedelta(days=days)
    temp = df[np.logical_and((df["date"]>=start_day),(df["date"]<end_day))].copy() 
    
    # =============================================================================
    # 计算数据集中相邻数据差的绝对值的环比
    # =============================================================================
    #前后相邻两个点做差
    temp.loc[:,"residual"] = temp.loc[:,"load"].diff().copy()
    #取绝对值
    temp.loc[:,"residual"] = temp.loc[:,"residual"].abs().copy()
    #求环比
    ratio = pd.DataFrame()
    
    residual = temp.loc[:,"residual"].to_frame()
    residual = residual.iloc[1:,:]
    residual = residual.reset_index(drop=True)
    
    actual = temp.loc[:,"load"].to_frame()
    actual = actual.iloc[:-1,:]
    actual = actual.reset_index(drop=True)
    
    ratio.loc[:,"ratio"] = (residual.loc[:,"residual"]/actual.loc[:,"load"]).abs()

    # =============================================================================
    # 画出环比的分布图
    # =============================================================================
    #设置画布大小
    plt.figure(figsize=(18,9))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    #画图
#    x = range(len(ratio))
    date = temp.loc[:,"date"].to_frame().copy()
    x = date.iloc[1:,:]
    plt.plot(x, ratio["ratio"],label="ratio",alpha=0.6)
    
    plt.legend(loc="upper left") 
    plt.title("{}-{}环比分布".format(filename[:11],start_day.strftime("%Y-%m-%d")))
    plt.savefig("./环比图/{}.png".format(filename[:-4]))  
    #plt.close()
    plt.show()




def merge_load_tmp(file="吴川站220kV#1主变-高有功值_15min_data.csv"):
    
    """
    函数功能
    将指定母线和天气数据进行合并
    """
    weather = pd.read_csv("./data/haikou_weather_15min.csv")
    weather["time"] = pd.to_datetime(weather["time"])
    
    df = pd.read_csv("./data/zhanjiang_sample_220_15mins_no_null/{}".format(file))
    df.columns = ["date","load"]
    df["date"] = pd.to_datetime(df["date"])
                     
    df2 = pd.merge(left = df,right=weather,how="inner",left_on="date",right_on="time")              
    df2 = df2.drop(columns="time")
    
    return df2

def create_dir_by_name(name):
    """
    函数功能：根据预测地区的名字，创建./area/name的文件目录。

    Parameters
    ----------
    name : str
        建立文件夹的名称.

    Returns
    -------
    None.

    """
    # 根目录的绝对路径
    absolute_path = "."

    if not os.path.exists(absolute_path + "/figures/"):
        os.makedirs(absolute_path + "figures/")

    if not os.path.exists(absolute_path + name + "/figures/"):
        os.makedirs(absolute_path + name + "/figures/" )
    
    dir_list = ["年趋势图","周趋势图","按天采样趋势图","指定日期图","连续周期图","环比图"]
    
    for dir_name in dir_list:
        #判断需要创建的路径是否存在，若不存在则创建
        if not os.path.exists(absolute_path + "/figures/" + name + "/{}".format(dir_name)):
            os.makedirs(absolute_path + "/figures/" + name + "/{}".format(dir_name))


def plot_distribution(df, feature_col, date_col, save_name, save_path="./"):
    """
    画出输入dataframe中指定特征列的数据分布情况

    Parameters
    ----------
    df : DataFrame
        待处理数据.
    feature_col : str
        待处理列名.
    date_col : str
        时间列名称.
    save_name : str
        保存图片的文件名.
    save_path : str, optional
        保存图片的路径 The default is "./".

    Returns
    -------
    None.

    """
    #将时间列准换成datetime格式
    df[date_col] = pd.to_datetime(df[date_col])
    # 画出数据的原始分布
    # sns.displot(df["tmp"])
    # 画图进行对比,设置画布大小
    plt.figure(figsize=(20, 10))

    # 解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False

    # 折线图画图
    fig, axes = plt.subplots(3, 1, figsize=(30, 30))
    ax1 = axes[0]
    ax1.plot(df[date_col], df[feature_col], label=feature_col, alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_xlabel("date")
    ax1.set_title("comparison")

    # 画直方分布图，用seaborn画的更加好看
    ax2 = axes[1]
    # sns.histplot(df[feature_col], ax=ax2)
    ax2.hist(df[feature_col], bins=50)

    # 画出数据和标准正太分布之间的差距
    ax3 = axes[2]
    ax3.set_title("normal distribution")
    prob = stats.probplot(df[feature_col], dist=stats.norm, plot=ax3)

    # 保存图片
    # fig.savefig(os.path.join(save_path, f"{save_name}_{feature_col}.png"), dpi=200)
    plt.show()
    plt.close("all")



###########################################画图预测表现及天气变化########################


import seaborn as sns
sns.set(style='whitegrid')


def mape(y_true, y_pred):
    """
    计算 mape

    Parameters
    ----------
    y_true : ndarry
        true label of prediction data.
    y_pred : ndarray
        predicted label of predictoin data.

    Returns
    -------
    TYPE
        the mape accuracy of predicted and actual .

    """
    return 1 - np.mean(np.abs((y_pred - y_true) / y_true))


def get_plt_elements(df,
                     time_col,
                     actual_col='actual',
                     pred_col='predict_load',
                     weather_col='tmp'):
    """计算每日准确率信息并返回

    Parameters
    ----------
    df : DataFrame 含有预测值与真实值和天气数据，预测结果数据集
    time_col : str 时间列列名
    actual_col : str, optional
        实测值列名, by default 'actual'
    pred_col : str, optional
        预测值列名, by default 'predict_load'
    weather_col : str, optional
        天气数据列名, by default 'tmp'

    Returns
    -------
    None 画图展示
    """

    #计算每日准确率
    start_date = df.loc[0, time_col].date()
    day_len = len(df) // 96
    date_index = pd.date_range(start=start_date, periods=day_len, freq='1D')
    acc_values = []
    for i in range(day_len):
        acc_mean = mape(df.loc[i * 96:(i + 1) * 96-1, actual_col].values,
                        df.loc[i * 96:(i + 1) * 96-1, pred_col].values)
        acc_values.append(acc_mean)
    day_acc_df = pd.DataFrame()
    day_acc_df['date'] = date_index
    day_acc_df['accuracy'] = acc_values
    day_acc_df['date'] = day_acc_df['date'].apply(
        lambda x: x + pd.Timedelta(hours=12))

    acc_dates = day_acc_df['date'].to_list()
    acc_vals = day_acc_df['accuracy'].to_list()

    tmp_max = df[weather_col].max()
    tmp_min = df[weather_col].min()
    scaler = (tmp_max - tmp_min) * 0.8

    return acc_dates, acc_vals, tmp_max, tmp_min, scaler


def plot_peroid2(df,
                fig_title=None,
                time_col="datetime",
                pred_col='predict_load',
                actual_col='actual',
                weather_col='tmp',
                start_day="2021-11-12",
                end_day=None,
                days=30):
    """按时间画出预测与真实值和天气，并计算每日准确率

    Parameters
    ----------
    df : DataFrame 需要画图的数据集
    fig_title : str, optional
        图形的title, by default None
    time_col : str, optional
        时间列列名, by default "datetime"
    pred_col : str, optional
        预测列列名, by default 'predict_load'
    actual_col : str, optional
        实际值列名, by default 'actual'
    weather_col : str, optional
        天气列列名, by default 'tmp'
    start_day : str, optional
        画图开始时间, by default "2021-11-12"
    end_day : str, optional
        画图截止时间, by default None
    days : int, optional
        画图天数与end_day二选一, by default 30
    """
    #将cols列全部转换成数值类型
    plt_cols = [actual_col, pred_col, weather_col]
    for col in plt_cols:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')

    #时间列转化成时间格式
    df[time_col] = pd.to_datetime(df[time_col])
    start_day = pd.to_datetime(start_day)
    if end_day is None:
        end_day = start_day + pd.Timedelta(days=days)
    else:
        end_day = pd.to_datetime(end_day)

    #画图进行对比,设置画布大小
    fig = plt.figure(figsize=(30, 10))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False

    #取需要画图的时间段数据
    temp = df[np.logical_and((df[time_col] > start_day),
                             (df[time_col] <= end_day))]
    temp = temp.reset_index(drop=True)

    #获取画图需要的准确率等信息
    acc_dates, acc_vals, weather_max, weather_min, scaler = get_plt_elements(
        df=temp,
        time_col=time_col,
        actual_col=actual_col,
        pred_col=pred_col,
        weather_col=weather_col)
    # 生成1x1的图位置为1
    ax1 = fig.add_subplot(111)
    if fig_title is not None:
        plt.title(fig_title)
    # 负荷的真实值和预测值曲线
    ax1.plot(temp[time_col], temp[actual_col], label='actual')
    ax1.plot(temp[time_col], temp[pred_col], label='predict')
    ax1.set_ylabel('load/负荷:(MW)')
    ax1.legend(loc='upper left')
    # 两个y轴共用x时间轴轴
    ax2 = ax1.twinx()
    # 天气曲线
    ax2.plot(temp[time_col],
             temp[weather_col],
             label=weather_col,
             color='green',
             linestyle='--')
    # 生成每日准确率的点
    ax2.scatter(acc_dates, [i * scaler for i in acc_vals],
                label='accuracy(%)',
                c='red')
    ax2.set_ylabel(weather_col)
    ax2.legend(loc='lower right')
    # 为每日添加准确率的值
    for i in range(len(acc_dates)):
        plt.text(acc_dates[i], acc_vals[i] * scaler,
                 str(round(acc_vals[i] * 100, 2)))
    plt.show()
    plt.close()

def get_plt_elements_multi(df,
                     time_col,
                     actual_col='actual',
                     pred_cols=['predict_load'],
                     weather_col='tmp'):
    """计算每日准确率信息并返回

    Parameters
    ----------
    df : DataFrame 含有预测值与真实值和天气数据，预测结果数据集
    time_col : str 时间列列名
    actual_col : str, optional
        实测值列名, by default 'actual'
    pred_col : str, optional
        预测值列名, by default 'predict_load'
    weather_col : str, optional
        天气数据列名, by default 'tmp'

    Returns
    -------
    None 画图展示
    """

    #计算每日准确率
    start_date = df.loc[0, time_col].date()
    day_len = len(df) // 96
    date_index = pd.date_range(start=start_date, periods=day_len, freq='1D')

    day_acc_df = pd.DataFrame()
    day_acc_df['date'] = date_index
    day_acc_df['date'] = day_acc_df['date'].apply(
        lambda x: x + pd.Timedelta(hours=12))

    acc_values = []
    for pred_col in pred_cols:
        acc_value = []
        for i in range(day_len):
            acc_mean = mape(df.loc[i * 96:(i + 1) * 96-1, actual_col].values,
                            df.loc[i * 96:(i + 1) * 96-1, pred_col].values)
            acc_value.append(acc_mean)

        day_acc_df['accuracy'] = acc_value
        acc_vals = day_acc_df['accuracy'].to_list()
        acc_values.append(acc_vals)

    acc_dates = day_acc_df['date'].to_list()

    tmp_max = df[weather_col].max()
    tmp_min = df[weather_col].min()
    scaler = (tmp_max - tmp_min) * 0.8

    return acc_dates, acc_values, tmp_max, tmp_min, scaler

def plot_peroid_multi(df,
                        fig_title=None,
                        time_col="datetime",
                        pred_cols=['predict_load'],
                        actual_col='actual',
                        weather_col='tmp',
                        start_day="2021-11-12",
                        end_day=None,
                        days=30):
    """按时间画出预测与真实值和天气，并计算每日准确率

    Parameters
    ----------
    df : DataFrame 需要画图的数据集
    fig_title : str, optional
        图形的title, by default None
    time_col : str, optional
        时间列列名, by default "datetime"
    pred_col : str, optional
        预测列列名, by default 'predict_load'
    actual_col : str, optional
        实际值列名, by default 'actual'
    weather_col : str, optional
        天气列列名, by default 'tmp'
    start_day : str, optional
        画图开始时间, by default "2021-11-12"
    end_day : str, optional
        画图截止时间, by default None
    days : int, optional
        画图天数与end_day二选一, by default 30
    """
    colors = ["red", "yellow", "blue", "black"]
    #将cols列全部转换成数值类型
    plt_cols = [actual_col, weather_col] + pred_cols
    for col in plt_cols:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')

    #时间列转化成时间格式
    df[time_col] = pd.to_datetime(df[time_col])
    start_day = pd.to_datetime(start_day)
    if end_day == None:
        end_day = start_day + pd.Timedelta(days=days)
    else:
        end_day = pd.to_datetime(end_day)

    #画图进行对比,设置画布大小
    fig = plt.figure(figsize=(30, 10))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False

    #取需要画图的时间段数据
    temp = df[np.logical_and((df[time_col] > start_day),
                             (df[time_col] <= end_day))]
    temp = temp.reset_index(drop=True)
    if not len(temp):
        raise Exception(f"select time from {start_day} to {end_day}, "
                        f"but data start time is from {df.loc[df.index[0], time_col]} to {df.loc[df.index[-1], time_col]}")

    #获取画图需要的准确率等信息
    acc_dates, acc_vals, weather_max, weather_min, scaler = get_plt_elements_multi(
        df=temp,
        time_col=time_col,
        actual_col=actual_col,
        pred_cols=pred_cols,
        weather_col=weather_col)
    # 生成1x1的图位置为1
    ax1 = fig.add_subplot(111)
    if fig_title is not None:
        plt.title(fig_title)
    # 负荷的真实值和预测值曲线
    ax1.plot(temp[time_col], temp[actual_col], label='actual')
    for pred_col in pred_cols:
        ax1.plot(temp[time_col], temp[pred_col], label=pred_col)
    ax1.set_ylabel('load/负荷:(MW)')
    ax1.legend(loc='upper left')
    # 两个y轴共用x时间轴轴
    ax2 = ax1.twinx()
    # 天气曲线
    ax2.plot(temp[time_col],
             temp[weather_col],
             label=weather_col,
             color='green',
             linestyle='--')
    # 生成每日准确率的点
    for index, pred_col in enumerate(pred_cols):
        acc_val = acc_vals[index]
        ax2.scatter(acc_dates, [i * scaler for i in acc_val],
                    label=f'{pred_col} accuracy(%)',
                    c=colors[index])
        for i in range(len(acc_dates)):
            plt.text(acc_dates[i], acc_val[i] * scaler,
                     str(round(acc_val[i] * 100, 2)))
    ax2.set_ylabel(weather_col)
    ax2.legend(loc='lower right')
    # 为每日添加准确率的值

    plt.show()
    plt.close()

        
# if __name__ == "__main__":
    
#     filename = "tongdiaozong_210615"
#     df = pd.read_csv("area/zhejiang_quanshehui_210628/data/history_cleaned_load.csv")
    
#     create_dir_by_name(filename)    
#     # plot_years(df,filename,start_day = "2018-1-1",years = 3)
#     plot_weeks(df,filename,col="load",start_day = "2020-6-1", weeks = 4)
#     # plot_sync(df,filename,start_day = "2018-1-1",years = 3)
#     # plot_peroid(df,filename,col="tmp",start_day = "2021-6-12",days = 15)
#     # plot_series(df,filename,start_day = "2021-5-24",period = 30)
#     # plot_ratio(df,filename,start_day = "2020-1-1",days =30 )

    
if __name__ == '__main__':

    df_result = pd.read_csv(
        './area/shaoxing_quanshehui/result/lstm_test_result.csv')
    df_result['datetime'] = pd.to_datetime(df_result['datetime'])
    df_weather = pd.read_csv(
        './area/shaoxing_quanshehui/data/history_cleaned_weather.csv')
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    plt_df = pd.merge(left=df_result,
                      right=df_weather,
                      how='left',
                      left_on='datetime',
                      right_on='date').reset_index(drop=True)
    plot_peroid2(plt_df, 'datetime', start_day='2021-12-14', days=30)
    
    
    

    #画出负荷曲线看下
    #path = "./data/zhanjiang_sample_220_15mins_no/"
    # path = "area/zhejiang_quanshengkoujian/data/history_cleaned_load.csv"
    # file_list = os.listdir(path)
    
    # for i,filename in enumerate(file_list):
        
        #筛选一下画出前5个数据
#        if (i<100) and (filename[-4:] == ".csv"):
        # if (i<100) and (filename[:3] == "椹北站"):
#        if (i<100) and (filename == "椹北站_sum.csv"):
            # pass;
        # else:
            # continue;
        #打印正在处理的站名
        # print("正在处理：{}".format(filename))
        
        # file_path = path + filename
        
        # df = pd.read_csv(file_path)
        
#        plot_years(df,filename,start_day = "2019-1-1",years = 1)
#        plot_weeks(df,filename,start_day = "2020-11-2", weeks = 3)
#        plot_sync(df,filename,start_day = "2018-1-1",years = 3)
#        plot_peroid(df,filename,start_day = "2019-5-19",days = 5)
#        plot_series(df,filename,start_day = "2019-5-18",period = 30)
        # plot_ratio(df,filename,start_day = "2020-1-1",days =30 )














