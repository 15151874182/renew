# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:04:56 2022

@author: zhangqipei
"""
"""
编写控制逻辑，主函数入口

"""

##python main.py --download --unzip --upload

import os
import sys

config_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.abspath(os.path.dirname(config_path))

if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"add project_path:{project_path} to sys path")

import datetime
import pandas as pd
import numpy as np
import traceback

# from sz_gansu.feature import data_generator
from config import config_parser, global_config, area_list
from database import download_daily, upload_daily, upzip_daily
from dataclean import clean_station, concat_base
# from sz_gansu.wind_prediction import station_predict
from train import run_predict
# from database_interface import DataInterface

#tmp_path = os.path.dirname(os.path.realpath(__file__))
# work_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

log_file = "logs/error.log"


def write_log(word):
    print(word)
    with open(log_file, "a") as f:
        f.write(f"{word}/n")


def check_predict(data, name):
    _len = len(data[data["load"] < 0])
    if _len > 0:
        write_log(f"{name}<0:{_len}")
    _nan = sum(data["load"].isnull())
    if _nan > 0:
        write_log(f"NAN {name} :{_nan}")
    # if len(data)%96!=0:
    #     write_log(f"{name}  shape error:{len(data)}, check!!")

    data["_date"] = pd.to_datetime(data["date"]).apply(lambda x: x.date())
    data_group = data.groupby("_date")
    for _date, day_data in data_group:
        if day_data.shape[0] == 96:
            continue
        print(f"{name} date:{_date} shape error:{day_data.shape}, check!!")


# In[]


def predict_process(stations, start_date, download=True, upload=True):
    """
    注意：时间范围前闭后开
    预测流程，若download=True,从ftp下载数据，转换格式，拼接到本地base，制作预测时间范围数据，
    调用预测函数，输出预测结果。

    Parameters
    ----------
    station : str
        场站名称.
    start_date : str
        "2022-8-26".
    end_date : str
        2022-8-26.

    Returns
    -------
    station : str
        场站名称.
    """
    # 时间范围前闭后开
    start_date = pd.to_datetime(start_date)

    # 根据站名，实例化配置文件
    config = config_parser(stations[0])
    date_col = config.get_para("time_name")

    # 从ftp取数据，然后转换成csv格式并保存到本地
    date_id = start_date.strftime("%Y%m%d")

    other_save_dir = os.path.join(global_config.work_path, 'data', "predict_output")
    print(f"other_save_dir:{other_save_dir}")

    # 获取的时候压缩包是昨天的，里面对应的日期是今天
    fetch_date_id = (start_date - pd.Timedelta(days=1)).strftime("%Y%m%d")

    # if download:
    #     for station in stations:
    #         fetch_start = start_date + pd.Timedelta(days=-10)
    #         fetch_end = start_date + pd.Timedelta(days=2)
    #         station_data_agent = DataInterface(station)
    #         station_data_agent.loadset(fetch_start, fetch_end)
    #         station_data_agent.weatherset(fetch_start, fetch_end)

    for station in stations:
        # 调用预测数据生成模块，生成指定时间端的预测输入数据
        try:
            config = config_parser(station)
            print(f"config.station_path:{config.station_path}")

            ####clean and concat nwp_weather
            daily_weather_path = os.path.join(config.station_path, "data", "daily_weather_from_db.csv")
            if not os.path.exists(daily_weather_path):
                print(f"can't find {station} daily_weather_from_db, skip")
                continue
            nwp_daily_weather = pd.read_csv(daily_weather_path)
            nwp_daily_weather, del_info = clean_station(config, use_type="test", df=nwp_daily_weather)
            nwp_daily_weather.drop_duplicates(subset=["date"], keep="last", inplace=True)
            nwp_daily_weather.to_csv(os.path.join(config.station_path, "data", "daily_cleaned_pred_weather.csv"),
                                      index=False)
            concat_base(config, new_df=nwp_daily_weather, data_type="nwp", data_name="weather")
            
            ####clean and concat real_load
            daily_real_load_path = os.path.join(config.station_path, "data", "daily_load_from_db.csv")
            if not os.path.exists(daily_real_load_path):
                print(f"can't find {station} daily_load_from_db, skip")
                continue
            daily_real_load = pd.read_csv(daily_real_load_path)
            daily_real_load, del_info = clean_station(config, use_type="test", df=daily_real_load,set_cols=['load'])
            daily_real_load.drop_duplicates(subset=["date"], keep="last", inplace=True)
            daily_real_load.to_csv(os.path.join(config.station_path, "data", "daily_cleaned_real_load.csv"),
                                      index=False)
            concat_base(config, new_df=daily_real_load, data_type="real", data_name="load")

            # 开始预测
            output = run_predict(station)

            result_path = os.path.join(config.get_para("station_path"), "result", "daily_mode_xgb_result.csv")
            output.to_csv(result_path, index=False)
        except:
            traceback.print_exc()
            print(f'{station} cant predict')

    if upload:
        for station in stations:
            station_data_agent = DataInterface(station)
            result_file = os.path.join(station_data_agent.config.get_para("station_path"), "result", "daily_mode_xgb_result.csv")
            pred_result = pd.read_csv(result_file)
            pred_result['date'] = pd.to_datetime(pred_result['date'])
            station_data_agent.insert_load_pred(pred_result)


