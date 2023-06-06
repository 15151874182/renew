# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:16:11 2022

@author: zhangqipei
"""

"""
定时器

"""

import os
import time
import datetime

while True:
    #获取当前的时间
    now = datetime.datetime.now()
    #设置启动时间
    today = datetime.datetime.strftime(now, "%Y/%m/%d")  #str格式
    set_time = today + " 06:00"
    #set_time = today + " 17:56"
    #转化为date time格式
    set_time = datetime.datetime.strptime(set_time, "%Y/%m/%d %H:%M")

    #判断当前时间是否达到程序启动时间
    if now > set_time and now < set_time + datetime.timedelta(minutes=5):
        print(f"start program at {now}")
        os.system(f"python main.py --station_name all --download --unzip --upload --start_date {now.date()}")
        #程序成功运行后，sleep 20mins
        time.sleep(1200)
    else:
        #暂停3mins后再次检测触发时间是否达到
        time.sleep(180)
