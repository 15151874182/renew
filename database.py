# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:08:17 2022

@author: zhangqipei
"""

"""
深圳公司提供的是FTP，这边编写FTP数据下载的函数接口

"""

import os
import copy
import datetime
import zipfile
import pandas as pd
from ftplib import FTP
from tqdm import tqdm
import sys

config_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.abspath(os.path.dirname(config_path))
if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"add project_path:{project_path} to sys path")


from config import global_config, station_info_get
from config import station_info as gansu_station_info

from center.tools.logger_function import get_logger
from center.tools.common import divide_by_96, del_file
logger = get_logger()

# In[]

#### 基础FTP连接、上传、下载
def ftpconnect(host, port, username, password):
    """
    根据ip和端口，用户和密码创建ftp连接对象

    Parameters
    ----------
    host : str
        IP.
    port : int
        端口.
    username : str
        用户账号.
    password : str
        用户密码.

    Returns
    -------
    ftp : TYPE
        ftp连接对象.

    """
    ftp = FTP()
    # ftp.set_debuglevel(2)         # 打开调试级别2，显示详细信息
    ftp.connect(host, port)        # 连接
    ftp.login(username, password)  # 登录，如果匿名登录则用空串代替即可
    return ftp


def download_file(ftp_config, remotepath, extract_path):
    """
    1、下载ftp服务器中的数据并保存到默认的本地路径
    2、将数据解压到本地默认的路径中

    Parameters
    ----------
    ftp_config : dict
        包含有ftp连接信息的字典对象.
        ftp_config = {"host": "120.76.75.9",
                      "port": 212,
                      "username": "GansuNariDown",
                      "password": "sdoklsj@1ss"}

    remotepath : str
        远程数据所在的路径，精确到文件名.
    extract_path : str
        本地数据解压路径.

    Returns
    -------
    None.

    """

    # 下载后原始压缩文件保存路径
    Zipdata_path = os.path.join(global_config.work_path, 'data', 'ftp_data', "download")
    if not os.path.exists(Zipdata_path):
        os.makedirs(Zipdata_path)

    # 解压后的数据路径
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # 下载zip压缩数据
    host = ftp_config["host"]
    port = ftp_config["port"]
    username = ftp_config["username"]
    password = ftp_config["password"]
    ftp = ftpconnect(host, port, username, password)

    # 压缩文件本地保存文件路径
    localpath = os.path.join(Zipdata_path, remotepath)
    # 下载数据并保存
    downloadfile(ftp, remotepath=remotepath, localpath=localpath)

    # 防止文件夹中存在其他数据，先清空文件夹再解压
    # del_file(filepath=extract_path)
    # 将数据解压到指定路径
    zip2file(zip_file_name=localpath, extract_path=extract_path)

# 下载文件
def downloadfile(ftp, remotepath, localpath):
    """
    在ftp上，下载指定文件

    Parameters
    ----------
    ftp : TYPE
        ftp连接对象.
    remotepaths : str
        远程的路径(需要精确到文件名)
    localpaths : str
        本地文件保存的路径(需要精确到文件名)

    Returns
    -------
    None.

    """
    # remotepath：上传服务器路径；localpath：本地路径；
    bufsize = 1024                # 设置缓冲块大小
    fp = open(localpath,'wb')    # 以写模式在本地打开文件
    
    #防止取不到数据但是创建多余空文件
    try:
        ftp.retrbinary('RETR ' + remotepath, fp.write, bufsize) # 接收服务器上文件并写入本地文件
        ftp.set_debuglevel(0)         # 关闭调试
        fp.close()                    # 关闭文件
        print(f"download {remotepath} successful!")
        return True
    except:
        fp.close()                    # 关闭文件
        os.remove(localpath)
        print(f"download failed : {remotepath}")
        return False


# 上传文件
def uploadfile(ftp, remotepath, localpath):
    """
    在指定的ftp上，上传本地文件到指定路径

    Parameters
    ----------
    ftp : TYPE
        ftp连接对象.
    remotepath : str
        远程的路径，需要精确到文件名.
    localpath : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    bufsize = 1024
    fp = open(localpath, 'rb')
    ftp.storbinary('STOR ' + remotepath, fp, bufsize)    # 上传文件
    ftp.set_debuglevel(0)
    fp.close()


#### 文件压缩、解压
# 压缩
def file2zip(zip_file_path, file_dir):
    """
    将指定路径的所有文件，压缩到一个zip包，并按名称保存

    Parameters
    ----------
    zip_file_path : str
        压缩后保存的zip文件绝对路径.
    file_dir : str
        txt文件所在文件夹的路径.

    Returns
    -------
    None.

    """
    file_names = os.listdir(file_dir)
    with zipfile.ZipFile(zip_file_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in file_names:
            zf.write(file_dir+"/"+fn, arcname=fn)


# 解压
def zip2file(zip_file_name, extract_path, members=None, pwd=None):

    with zipfile.ZipFile(zip_file_name) as zf:
        zf.extractall(extract_path, members=members, pwd=pwd)


#### 下载NWP数据格式转换




def transfer_wind_weather(dir_path, file):
    """
    处理风电站的nwp数据格式,处理单个txt文件
    对ftp下载下来的天气预报数据，解压后从txt转换到csv格式

    Parameters
    ----------
    dir_path : str
        文件夹路径.
    file : str
        文件名.

    Returns
    -------
    None.

    """

    # 建立列灭那个的中英文映射关系
    features = ['风速', '风向', '温度', '湿度', '气压']
    map_vals = ['speed', 'dir', 'temp', 'hum', 'press']
    feature_map = {key: val for key, val in zip(features, map_vals)}

    # print(f"processing {file}")
    # 读数据基本列处理
    df = pd.read_csv(os.path.join(dir_path, file), header=None)
    df.columns = ['年月日', '时分秒', '层高', '风速', '风向', '温度', '湿度', '气压']
    df['date'] = df['年月日']+' ' + df['时分秒']
    df.drop(columns=['年月日', '时分秒'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    # 分组处理合并
    group = df.groupby('层高')
    height_select = list(set(df['层高']))
    result_df = pd.DataFrame()
    # 遍历处理所有不同层高下的所有数据数据
    for height in height_select:
        height_data = copy.deepcopy(group.get_group(height))
        height_data.drop(columns=['层高'], inplace=True)
        for col in features:
            height_data.rename(
                columns={col: f'{feature_map[col]}_{height}'}, inplace=True)
        if result_df.empty:
            result_df = height_data
        else:
            result_df = pd.merge(
                left=result_df, right=height_data, on='date', how='left')

    # 删除时间戳重复数据
    result_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    # 对数据按时间戳进行排序
    result_df.sort_values(by='date', kind='mergsort', ascending=True)
    # 排序后的index重置
    result_df.reset_index(drop=True, inplace=True)

    # 补全15分钟间隔的数据
    start_time = result_df.loc[0, 'date']
    end_time = result_df.loc[result_df.shape[0]-1, 'date']
    time_df = pd.date_range(start=start_time, end=end_time,
                            freq='15T').to_frame(name='date')
    result_df = pd.merge(left=time_df, right=result_df, on='date', how='left')
    # index重置
    result_df.reset_index(drop=True, inplace=True)

    return result_df


def transfer_power(file_path, gys):
    """
    处理供应商提供的预报功率数据，XBY or XXL，处理单个txt文件
    处理成dataframe格式，并补全时间戳，未填充

    Parameters
    ----------
    file_path : str
        文件路径.
    gys : str
        供应商名称，用于生成power列时使用.

    Returns
    -------
    df : DataFrame
        处理后的数据.

    """
    print(f"prcessing:{file_path}")
    gys_name = gys
    
    try :
        df = pd.read_csv(r"{}".format(file_path))
    #如果数据是空的则直接返回
    except  pd.io.parsers.EmptyDataError:
        return pd.DataFrame()
        
    #删除数据中存在的nan
    df.dropna(how='any',axis=0,inplace=True)
    if len(df) == 0:
        return pd.DataFrame()
    
    #创建时间列
    df['date'] = df['Date']+' ' + df['Time(Asia/Shanghai)']
    df['date'] = pd.to_datetime(df['date'])
    df.drop(columns=['Date', 'Time(Asia/Shanghai)'], inplace=True)
    
    #重命名功率列名，附上对应供应商名字
    df.rename(columns={"Power(kW)":f"load_{gys_name}"},inplace=True)
    
    # 删除时间戳重复数据
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    # 对数据按时间戳进行排序
    df.sort_values(by='date', kind='mergsort', ascending=True)
    # 排序后的index重置
    df.reset_index(drop=True, inplace=True)
    
    # 补全15分钟间隔的数据,从第一个00：00开始截取
    start_time = df['date'].iloc[0]
    start_time_zero = str(start_time.date()) + " 00:00"
    start_time_zero = pd.to_datetime(start_time_zero)
    if start_time == start_time_zero:
        pass;
    else:
        start_time = start_time_zero + pd.Timedelta(days=1)
    
    #保证end_time,结尾为23:45
    end_time = df['date'].iloc[-1]
    end_time = str(end_time.date()) + " 23:45"
    end_time = pd.to_datetime(end_time)
    
    time_df = pd.date_range(start=start_time, end=end_time,
                            freq='15T').to_frame(name='date')
    df = pd.merge(left=time_df, right=df, on='date', how='left')
    # index重置
    df.reset_index(drop=True, inplace=True)
    
    return df

def transfer_solar_weather(dir_path, file):
    """
    处理光伏站的nwp数据格式,处理单个txt文件
    对ftp下载下来的天气预报数据，解压后从txt转换到csv格式

    Parameters
    ----------
    dir_path : str
        文件夹路径.
    file : str
        文件名.

    Returns
    -------
    None.

    """
    # txt数据原始列名
    en_name = ['年月日',
               '时分秒',
               'HGRAD',
               'HSRAD',
               'VDRAD',
               'HGRADVI',
               'HSRADVI',
               'VDRADVI',
               'COSZ',
               'TEMP',
               'HUM',
               'PRESSURE',
               'V1SP',
               'V1DIR']
    # 对应的中文列名
    ch_name = ['年月日',
               '时分秒',
               '全辐射',
               '散辐射',
               '直辐射',
               '可见光全辐射',
               '可见光散辐射',
               '可见光致辐射',
               '太阳天顶角余弦',
               '温度',
               '相对湿度',
               '地面气压',
               '风速',
               '风向']

    # print(f"processing {file}")
    df = pd.read_csv(os.path.join(dir_path, file), header=None)
    try:
        # 重命名列名
        df.columns = en_name
        # 将全辐射、直接负荷、温度列进行重命名，保证和训练集中的列名对应起来
        df.rename(columns={"HGRAD": "GlobalR", "VDRAD": "DirectR",
                  "TEMP": "AirT", "HUM": "RH"}, inplace=True)
    except:
        import traceback
        traceback.print_exc()
        print(f'rename error file_name:{file}')

    df['date'] = df['年月日']+' ' + df['时分秒']
    df.drop(columns=['年月日', '时分秒'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # 删除时间戳重复数据
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    # 对数据按时间戳进行排序
    df.sort_values(by='date', kind='mergsort', ascending=True)
    # 排序后的index重置
    df.reset_index(drop=True, inplace=True)

    # 补全15分钟间隔的数据
    start_time = df.loc[0, 'date']
    end_time = df.loc[df.shape[0]-1, 'date']
    time_df = pd.date_range(start=start_time, end=end_time,
                            freq='15T').to_frame(name='date')
    df = pd.merge(left=time_df, right=df, on='date', how='left')
    # index重置
    df.reset_index(drop=True, inplace=True)

    # # 保存文件
    # file_name = file.replace(".txt", ".csv")
    # save_name = f"NARI_{file_name}"
    # df.to_csv(f'./transfered_gansu_weather/{save_name}', index=False)

    return df

def get_pred_input(dir_path, predict_input):
    """
    输入解压后的txt所在的文件夹路径，将txt数据转换成csv并保存到predict_input

    Parameters
    ----------
    dir_path : str
        解压后的txt文件，所在的文件夹路径.
    predict_input : str
        转换后csv文件保存的路径.

    Returns
    -------
    None.

    """

    # 创建处理成csv的文件，保存路径
    if not os.path.exists(predict_input):
        os.makedirs(predict_input)

    # 防止文件夹中存在其他数据，先清空文件夹
    # del_file(filepath=predict_input)

    # 解压后的txt文件，所在的文件夹路径
    # station_info_path = os.path.join(global_config.work_path, 'data', '预测天气、实测天气、实测功率站名交集表.xlsx')
    station_info_path = os.path.join(global_config.work_path, 'data', 'gansu_station_info.xlsx')
    station_info = pd.read_excel(station_info_path, engine="openpyxl")
    station_info.index = station_info["FarmCode"]
    FarmCodes = list(station_info["FarmCode"])
    # 创建一个dict，建立WFarmCode 到 station_type的映射
    # farmcode_type_dict = dict(
    #     zip(station_info["WFarmCode"], station_info["station_type"]))

    # 获取所有文件夹的名称
    file_list = os.listdir(dir_path)

    ab_list = []
    for file in tqdm(file_list): ##遍历download txt中所有的站
        file_name = file.split("-Nwp")[0]
        if file_name not in FarmCodes:##选出和station_info["FarmCode"]有交集的站
            print(f"can't find {file_name} in FarmCodes")
            ab_list.append(file)
            continue
        station_name = station_info.loc[file_name, "FarmCode"]
        station_type = station_info.loc[file_name, "StationType"]
        # FarmCode_list = file.split("-")[0:4]
        # station_name = FarmCode_list[3].upper()
        # FarmCode = "-".join(FarmCode_list)
        if station_type == "pv":
            df = transfer_solar_weather(dir_path, file)
            save_name = f"daily_GF_pred_weather.csv"

        else:
            df = transfer_wind_weather(dir_path, file)
            save_name = f"daily_FD_pred_weather.csv"

        if not os.path.exists(os.path.join(predict_input, station_name, "data")):
            print(f"can't find {station_name}")
            continue
        save_path = os.path.join(predict_input, station_name, "data", save_name)
        # 将数据转换成96整除
        df = divide_by_96(df)
        df.to_csv(save_path, index=False)

    print(f"\n 以下这些站无法建立对应关系：{ab_list}")

# def get_pred_input(dir_path, predict_input):
#     """
#     输入解压后的txt所在的文件夹路径，将txt数据转换成csv并保存到predict_input
#
#     Parameters
#     ----------
#     dir_path : str
#         解压后的txt文件，所在的文件夹路径.
#     predict_input : str
#         转换后csv文件保存的路径.
#
#     Returns
#     -------
#     None.
#
#     """
#
#     # 创建处理成csv的文件，保存路径
#     if not os.path.exists(predict_input):
#         os.makedirs(predict_input)
#
#     # 防止文件夹中存在其他数据，先清空文件夹
#     # del_file(filepath=predict_input)
#
#     # 解压后的txt文件，所在的文件夹路径
#     station_info_path = os.path.join(global_config.work_path, 'data', '预测天气、实测天气、实测功率站名交集表.xlsx')
#     station_info = pd.read_excel(station_info_path, engine="openpyxl")
#     # 创建一个dict，建立WFarmCode 到 station_type的映射
#     farmcode_type_dict = dict(
#         zip(station_info["WFarmCode"], station_info["station_type"]))
#
#     # 获取所有文件夹的名称
#     file_list = os.listdir(dir_path)
#
#     ab_list = []
#     for file in tqdm(file_list):
#         FarmCode_list = file.split("-")[0:4]
#         station_name = FarmCode_list[3].upper()
#         FarmCode = "-".join(FarmCode_list)
#         try:
#             station_type = farmcode_type_dict[FarmCode]
#         except:
#             ab_list.append(file)
#             continue
#         if station_type == "光伏":
#             df = transfer_solar_weather(dir_path, file)
#             save_name = f"daily_GF_pred_weather.csv"
#
#         else:
#             df = transfer_wind_weather(dir_path, file)
#             save_name = f"daily_FD_pred_weather.csv"
#
#         save_path = os.path.join(predict_input, station_name, "data", save_name)
#         # 将数据转换成96整除
#         df = divide_by_96(df)
#         df.to_csv(save_path, index=False)
#
#     print(f"\n 以下这些站无法建立对应关系：{ab_list}")


#### 预测csv数据转换到txt
# def transfer_csv_to_txt(read_dir, file_name, save_dir, date_id):
#     """
#     函数需要读取文件 ”预测天气、实测天气、实测功率站名交集表.xlsx“
#     读取指定文件路径下的csv文件，转换成txt格式，并保存到指定路径
#
#     Parameters
#     ----------
#     read_dir : str
#         读取的csv文件所在文件夹绝对路径.
#     filename : str
#         读取的csv文件名.
#     save_dir : str
#         输出保存的txt文件所在文件夹绝对路径.
#     date_id : str
#         使用的天气预报日期
#         now = datetime.datetime.now()
#         date_id = now.strftime("%Y%m%d").
#     Returns
#     -------
#     None.
#
#     """
#     # 根据文件名，获取子站名称
#     station_name = file_name.split("_")[1].upper()
#     station_info_path = os.path.join(global_config.work_path, 'data', '预测天气、实测天气、实测功率站名交集表.xlsx')
#     station_info = pd.read_excel(station_info_path, engine="openpyxl")
#     # 创建一个dict，建立station_name 到 WFarmCode的映射
#     station_name_farmcode_dict = dict(
#         zip(station_info["station_name"], station_info["WFarmCode"]))
#
#     # 构建保存成txt的文件名
#     # 获取每日下载的远端文件名
#     # date_id表示的是使用的nwp文件上传时间，因此生成的预测power文件日期要+1天
#     date_id = datetime.datetime.strptime(date_id, "%Y%m%d")
#     date_id = date_id + datetime.timedelta(days=1)
#     date_id = date_id.strftime("%Y%m%d")
#     save_name = station_name_farmcode_dict[station_name]
#
#     # 构建txt名称
#     save_name_full = f"{save_name}-short-{date_id}.txt"
#     # 将NARI替换成SGEPRI
#     save_name_full = save_name_full.replace("NARI", "SGEPRI")
#     save_path = os.path.join(save_dir, save_name_full)
#
#     # 读取预测出的csv文件，创建必要的列，修改列名
#     file_path = os.path.join(read_dir, file_name)
#     df = pd.read_csv(file_path)
#     df["date"] = pd.to_datetime(df["date"])
#     df["Date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
#     df["Time(Asia/Shanghai)"] = df["date"].apply(lambda x: x.strftime("%H:%M:%S"))
#     df["WFarmCode"] = save_name
#     df.rename(columns={"load": "Power(kW)"}, inplace=True)
#     df.drop(columns="date", inplace=True)
#     df = df.loc[:, ['WFarmCode', 'Date', 'Time(Asia/Shanghai)', 'Power(kW)']]
#
#     # 保存到本地txt文件
#     df.to_csv(save_path, index=False)

def transfer_csv_to_txt(read_dir, file_name, save_dir, date_id):
    """
    函数需要读取文件 ”预测天气、实测天气、实测功率站名交集表.xlsx“
    读取指定文件路径下的csv文件，转换成txt格式，并保存到指定路径
    :param read_dir:    读取的csv文件所在文件夹绝对路径.
    :param file_name:   读取的csv文件名.
    :param save_dir:    输出保存的txt文件所在文件夹绝对路径.
    :param date_id:     使用的天气预报日期
                        now = datetime.datetime.now()
                        date_id = now.strftime("%Y%m%d").
    :return:
    """
    # 根据文件名，获取子站名称
    # station_name = file_name.split("_")[1].upper()
    station_name = file_name.split("_")[1]
    # 构建保存成txt的文件名
    # 获取每日下载的远端文件名
    # date_id表示的是使用的nwp文件上传时间，因此生成的预测power文件日期要+1天
    date_id = datetime.datetime.strptime(date_id, "%Y%m%d")
    date_id = date_id + datetime.timedelta(days=1)
    date_id = date_id.strftime("%Y%m%d")
    # save_name = gansu_station_info.loc[station_name, "FarmCode"]
    save_name= station_name
    # 构建txt名称
    save_name_full = f"{save_name}-short-{date_id}.txt"
    # 将NARI替换成SGEPRI
    save_name_full = save_name_full.replace("NARI", "SGEPRI")
    save_path = os.path.join(save_dir, save_name_full)

    # 读取预测出的csv文件，创建必要的列，修改列名
    file_path = os.path.join(read_dir, file_name)
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df["Date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    df["Time(Asia/Shanghai)"] = df["date"].apply(lambda x: x.strftime("%H:%M:%S"))
    df["WFarmCode"] = save_name
    df.rename(columns={"load": "Power(kW)"}, inplace=True)
    df.drop(columns="date", inplace=True)
    df = df.loc[:, ['WFarmCode', 'Date', 'Time(Asia/Shanghai)', 'Power(kW)']]

    # 保存到本地txt文件
    df.to_csv(save_path, index=False)

def upload_daily(ftp_config, date_id):
    """
    将预测输出的csv转换成txt，并压缩成指定名称的zip包，然后上传到指定的ftp中

    Parameters
    ----------
    ftp_config : dict
        ftp_config = {"host": "120.76.75.9",
                      "port": 212,
                      "username": "GansuNariUpload",
                      "password": "sdhsih@67as"}
    date_id : str
        使用的天气预报日期
        now = datetime.datetime.now()
        date_id = now.strftime("%Y%m%d").

    Returns
    -------
    None.

    """

    # 读取预测输出predict_output文件夹内的csv，转换成txt
    read_dir = os.path.join(global_config.work_path, 'data', "predict_output")
    file_list = os.listdir(read_dir)
    save_dir = os.path.join(global_config.work_path, 'data', 'ftp_data', "daily_upload")

    # 创建转换后txt保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 防止文件夹中存在其他数据，先清空文件夹
    del_file(filepath=save_dir)

    for file_name in tqdm(file_list):
        transfer_csv_to_txt(read_dir, file_name, save_dir, date_id)

    # 将txt文件都压缩成一个zip包
    zip_file_name = f"Gansu_nari_power_06_{date_id}.zip"
    zip_file_path = os.path.join(global_config.work_path, 'data', 'ftp_data', zip_file_name)
    if os.path.exists(zip_file_path):
        print(f"{zip_file_path} exist, remove")
        os.remove(zip_file_path)
    file2zip(zip_file_path=zip_file_path, file_dir=save_dir)

    # 上传zip包到ftp
    host = ftp_config["host"]
    port = ftp_config["port"]
    username = ftp_config["username"]
    password = ftp_config["password"]
    # 链接
    ftp = FTP()
    ftp.set_pasv(True)
    ftp.connect(host, port)  # \u8fde\u63a5
    ftp.login(username, password)
    # ftp = ftpconnect(host, port, username, password)
    uploadfile(ftp, remotepath=zip_file_name, localpath=zip_file_path)


def _download_file_(ftp_config, remotepath):
    """
    1、下载ftp服务器中的数据并保存到默认的本地路径
    2、将数据解压到本地默认的路径中
    :param ftp_config: dict
        包含有ftp连接信息的字典对象.
    :param remotepath:  远程数据所在的路径，精确到文件名.
    :return:
    """

    # 下载后原始压缩文件保存路径
    Zipdata_path = os.path.join(global_config.work_path, 'data', 'ftp_data', "download")
    if not os.path.exists(Zipdata_path):
        os.makedirs(Zipdata_path)



    # 下载zip压缩数据
    host = ftp_config["host"]
    port = ftp_config["port"]
    username = ftp_config["username"]
    password = ftp_config["password"]
    ftp = ftpconnect(host, port, username, password)

    # 压缩文件本地保存文件路径
    localpath = os.path.join(Zipdata_path, remotepath)
    # 下载数据并保存
    return downloadfile(ftp, remotepath=remotepath, localpath=localpath)

def download_daily(ftp_config, date_id):
    """
    从ftp获取每日的数据，并转换格式输出到指定文件夹

    date_id : str
        使用的天气预报日期
        now = datetime.datetime.now()
        date_id = now.strftime("%Y%m%d").
    """
    # 远程文件路径
    file_weather6 = f"Gansu_weather_06_{date_id}.zip"
    file_weather18 = f"Gansu_weather_18_{date_id}.zip"

    # remotepath = [file_load,file_weather18]
    remotepath = file_weather6

    # 压缩文件解压后的txt保存路径
    extract_path = os.path.join(global_config.work_path, 'data', 'ftp_data', "daily_extract")
    # 最终输出csv的保存路径
    predict_input = os.path.join(global_config.work_path, 'data', "area")
    # _station_path = os.path.join(global_config.work_path, "data", "area")

    # 下载文件，并解压到指定文件
    is_download = _download_file_(ftp_config, remotepath)
    # download_file(ftp_config, remotepath, extract_path)
    if is_download:
        print(f"download file {remotepath} successful!")
    # 从解压后的路径读取txt文件，并转换成csv保存到指定路径
    # get_pred_input(dir_path=extract_path, predict_input=predict_input)

def upzip_daily(date_id):
    remotepath = f"Gansu_weather_06_{date_id}.zip"
    # 下载后原始压缩文件保存路径
    Zipdata_path = os.path.join(global_config.work_path, 'data', 'ftp_data', "download")
    # 压缩文件本地保存文件路径
    localpath = os.path.join(Zipdata_path, remotepath)
    # 压缩文件解压后的txt保存路径
    extract_path = os.path.join(global_config.work_path, 'data', 'ftp_data', "daily_extract")
    # 最终输出csv的保存路径
    predict_input = os.path.join(global_config.work_path, 'data', "area")
    if not os.path.exists(localpath):
        # print(f"{localpath} can't find")
        raise Exception(f"{localpath} can't find")
    # 解压后的数据路径
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    # 防止文件夹中存在其他数据，先清空文件夹再解压
    del_file(filepath=extract_path)
    # 1. 将数据解压到指定路径
    print('unziping....')
    zip2file(zip_file_name=localpath, extract_path=extract_path)
    # 2. 从解压后的路径读取txt文件，并转换成csv保存到指定路径
    get_pred_input(dir_path=extract_path, predict_input=predict_input)

    print(f"unzip {localpath} done.")

#### 业务逻辑，每日下载数据
def station_download(station, date_id, gys_list=["XXL","XBY"]):
    """
    从ftp获取指定日期数据，并转换格式输出到指定文件夹
    
    station : str
        需要下载的场站名称
    date_id : str
        使用的天气预报日期
        now = datetime.datetime.now()
        date_id = now.strftime("%Y%m%d").
    gys_list : list
        供应商名称，目前只支持["XXL","XBY"]
        
    """
    #转换场站的名称
    ID = station_info_get(station, name="id")
    area = station_info_get(station, name="area")
    station = f"NARI-{ID}-{area}-{station}"
    
    for gys in gys_list:
        #针对不同厂家的文件名路径做区分
        if gys == "XBY":
            power_name = f"{station}_short_{date_id}.txt"
        elif gys == "XXL":
            power_name = f"{station}-short-{date_id}.txt"
        # 远程文件路径   
        nwp_name = f"{station}-Nwp-{date_id}.txt"
        file_nwp = f"ShanDongGH/{gys}/{nwp_name}"
        file_power = f"ShanDongGH/{gys}/{power_name}"
        
        #ftp下载文件保存的目录
        localdir = f"./ftp_data/download_data/{gys}/{station}"
        if not os.path.exists(localdir):
            os.makedirs(localdir)
        
        #根据配置文件，读取ftp下载文件配置
        host = global_config.ftp_host
        port = global_config.ftp_port
        username = global_config.ftp_username
        password = global_config.ftp_password
        ftp = ftpconnect(host, port, username, password)
        
        #下载数据并保存到本地
        downloadfile(ftp, remotepath=file_nwp,localpath=f"{localdir}/{nwp_name}")
        downloadfile(ftp, remotepath=file_power,localpath=f"{localdir}/{power_name}")

def station_transfer(station, date_id, gys):
    """
    读取download_data目录，将FTP下载下来的txt转换成dataframe
    注意：如果数据由于格式问题转换失败，则返回空的DataFrame
    返回供应商提供的数值天气预报：nwp_data
                预报功率：power_data
                
    Parameters
    ----------
    station : str
        需要下载的场站名称.
    date_id : str
        使用的天气预报日期
        now = datetime.datetime.now()
        date_id = now.strftime("%Y%m%d")..
    gys : str
        只支持"XXL" or "XBY"

    Returns
    -------
    nwp_data : DataFrame
        供应商提供的数值天气预报.
    power_data : DataFrame
        供应商提供的预报功率.
    """
    
    #转换场站的名称
    ID = station_info_get(station, name="id")
    area = station_info_get(station, name="area")
    station = f"NARI-{ID}-{area}-{station}"
    
    #针对不同厂家的文件名路径做区分
    if gys == "XBY":
        power_name = f"{station}_short_{date_id}.txt"
    elif gys == "XXL":
        power_name = f"{station}-short-{date_id}.txt"
    
    nwp_name = f"{station}-Nwp-{date_id}.txt"
    file_nwp = f"./ftp_data/download_data/{gys}/{station}/{nwp_name}"
    file_power = f"./ftp_data/download_data/{gys}/{station}/{power_name}"
        
    if os.path.exists(file_nwp):
        nwp_data = transfer_wind_weather(file_nwp,gys)
    else:
        logger.debug(f"warning {gys} nwp data is empty!")
        nwp_data = pd.DataFrame()
        
    if os.path.exists(file_power): 
        power_data = transfer_power(file_power,gys)
    else:
        logger.debug(f"warning {gys} power data is empty!")
        power_data = pd.DataFrame()
        
    return nwp_data, power_data


def download_transfer(station,start_date,end_date,download=True):
    """
    输入站名，对起始时间和结束时间范围内的数据进行下载并转换成csv文件保存
    当 download = False 时，不下载数据，直接从本地读取txt文件

    Parameters
    ----------
    station : str
        站名，同配置文件中的站名，例如 sdghrc.
    start_date : datetime
        起始时间.
    end_date : datetime
        结束时间.
    download : bool, optional
        是否需要连接FTP下载数据. The default is True.
    
    """

    while start_date < end_date:
        print(f"processing : {start_date}")
        date_id = start_date.strftime("%Y%m%d")
        
        # 判断是否需要下载数据，不下载则直接从本地读取
        if download:
            # 从ftp下载数据到本地download_data
            station_download(station,date_id,gys_list=["XXL","XBY"])
        
        #读取本地download_data文件，将里面的txt转换成csv保存到transfered_data
        for gys in ["XXL","XBY"]:
            nwp_data,power_data = station_transfer(station,date_id,gys=gys)
            # 保存数据到transfered_data文件夹下
            if len(nwp_data) != 0:
                # 保存文件的名称
                nwp_name = f"{station}-Nwp-{date_id}.csv"
                # 保存文件目录路径
                save_dir = f"./ftp_data/transfered_data/{gys}/{station}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                nwp_data.to_csv(f"{save_dir}/{nwp_name}",index=False)
                
            if len(power_data) != 0:
                # 保存文件的名称
                power_name = f"{station}-short-{date_id}.csv"
                # 保存文件目录路径
                save_dir = f"./ftp_data/transfered_data/{gys}/{station}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                power_data.to_csv(f"{save_dir}/{power_name}",index=False)
            
        start_date = start_date + pd.Timedelta(days=1)
        

if __name__ == "__main__":
    
    # now = datetime.datetime.now() + pd.Timedelta(days=1)
    # date_id = now.strftime("%Y%m%d")
    
    # start_date = datetime.datetime(2022, 8,7)
    # end_date = datetime.datetime(2022, 8, 24)
    #
    # #批量转换csv并保存
    # for station in area_list:
    #     download_transfer(station,start_date,end_date,download=False)

    # extract_path = os.path.join(global_config.work_path, 'data', 'ftp_data', "daily_extract")
    # # 最终输出csv的保存路径
    # predict_input = os.path.join(global_config.work_path, 'data', "area")
    # # _station_path = os.path.join(global_config.work_path, "data", "area")
    #
    # # 下载文件，并解压到指定文件
    # # download_file(ftp_config, remotepath, extract_path)
    # # print("download file successful!")
    # # 从解压后的路径读取txt文件，并转换成csv保存到指定路径
    # get_pred_input(dir_path=extract_path, predict_input=predict_input)

    import argparse

    now = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=pd.to_datetime,
        default=now,  # "2022-1-7",#'2019-1-1',
        help="predict start date, default:tomorrow",
    )
    parser.add_argument(
        "--download",
        default=False,  # "2022-1-7",#'2019-1-1',
        action='store_true',
        help="download data or not",
    )
    parser.add_argument(
        "--upload",
        default=False,  # "2022-2-10",  # "2021-10-10",
        action='store_true',
        help="upload data or not ",
    )
    args = parser.parse_args()
    download = args.download
    upload = args.upload
    _date = pd.to_datetime(args.date)

    date_id = _date.strftime("%Y%m%d")

    if download:
        # 从ftp下载数据到本地download_data目录中
        print("download data ing")
        # ftp连接配置信息
        download_config = {"host": "120.76.75.9",
                           "port": 212,
                           "username": "GansuNariDown",
                           "password": "sdoklsj@1ss"}
        download_daily(ftp_config=download_config, date_id=date_id)

    if upload:
        from sz_gansu.config import global_config, area_list, config_parser
        all_area_list = os.listdir(os.path.join(global_config.work_path, "data", "area"))
        stations = [n for n in area_list if n in all_area_list]
        other_save_dir = os.path.join(global_config.work_path, 'data', "predict_output")
        for station in stations:
            config = config_parser(station)
            result_path = os.path.join(config.get_para("station_path"), "result", "daily_predict_result.csv")
            if not os.path.exists(result_path):
                continue
            _station_type = "GF" if config.get_para("station_type") == "pv" else "FD"
            result_path2 = os.path.join(other_save_dir,
                                       f"{_station_type}_{station}_real_weather_trainset_predict_result.csv")
            output = pd.read_csv(result_path)
            output.to_csv(result_path2, index=False)

        print("upload data ing")
        # ftp连接配置信息
        upload_config = {"host": "120.76.75.9",
                         "port": 212,
                         "username": "GansuNariUpload",
                         "password": "sdhsih@67as"}

        # upload_daily(ftp_config=upload_config, date_id=date_id)

        upload_config2 = {"host": "202.104.113.194",
                          "port": 2123,
                          "username": "GanSuNari_wr",
                          "password": "nari0755GanSuNari"}
        upload_daily(ftp_config=upload_config2, date_id=date_id)
    
    
    

    
    