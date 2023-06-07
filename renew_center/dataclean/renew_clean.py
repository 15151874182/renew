import os
import sys
import time
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore') 

from renew_center.tools import setup_logger
logger=setup_logger('logger')

class Clean():
    def __init__(self):
        pass
    def clean(self,data,selected_feas,
              area,
              area_type,
              use_type,  ##'train'(may delete bad feas)/'predcit'(fix data only)
              capacity,
              online,   ##None for wind clean
              longitude, ##None for wind clean
              latitude): ##None for wind clean

#preparing ====================================================================        
        day_point=96  ##15min time freq everyday
        df=deepcopy(data)
        df=df[selected_feas]
        if area_type=='pv' and online==True:
            from astral.sun import sun
            from astral import LocationInfo            
            #suntime是日出日落信息，需要联网
            self.suntime = self.download_suntime(area=area,
                                                belong='china',
                                                longitude=longitude,
                                                latitude=latitude,
                                                begin="2018-1-1",
                                                end="2024-12-31")
            self.suntime["sunrise_before"] = self.suntime["Sunrise"].apply(lambda x:self.sunrise_before(x))
            self.suntime["sunset_after"] = self.suntime["Sunset"].apply(lambda x:self.sunset_after(x))
            self.suntime["start_zero_flag"] = 0               
        elif area_type=='pv' and online==False:
            self.suntime = pd.read_csv('data/suntime.csv')
# =============================================================================             
        for col in selected_feas:
            #### fea name should follow rules as below
            if "speed" in col:    ##wind speed
                upper,lower = 35,0  
            elif "dir" in col:    ##wind direction
                upper,lower = 360,0  
            elif "rad" in col:    ##radiation
                upper,lower = 1400,0     
            elif "temp" in col: ##temperature   
                upper,lower = 48,-15  
            elif "hum" in col:  ##humidity
                upper,lower = 100,-0  
            elif "press" in col:##pressure    
                upper,lower = 1500,640  
            elif "load" in col:
                upper,lower = capacity,0  
            else:
                upper,lower=float('inf'),float('-inf')
            logging.info(f"{col},upper={upper},lower={lower}")   
#1.handle_NAN =================================================================  
            df=self.handle_NAN(df,col,threshold=day_point//6)
#2.handle_limit================================================================            
            df=self.handle_limit(df,col,lower,upper)
            if area_type=='wind':
                df=self.handle_constant(df,col,capacity,threshold=day_point//6)
                
            # if area_type=='wind' and use_type=='train':
            #     pass
            # elif area_type=='wind' and use_type=='predcit':
            #     pass
            # elif area_type=='pv' and use_type=='train':
            #     pass
            # elif area_type=='pv' and use_type=='predcit':
            #     pass
        
    def clean_area(self,config,selected_feas):
        return self.clean(data=config.data,
                          selected_feas=selected_feas,
                          use_type=config.use_type,  ##'train'/'predcit'
                          capacity=config.capacity,
                          longitude=config.longitude,
                          latitude=config.latitude)
    
    def check_length(self,data,day_point,threshold):
        days=len(data)//day_point
        ##return True if number of data is enough
        return True if days>=threshold else False

    def handle_NAN(self,data,col,threshold):
        df=deepcopy(data)
        #### spline interpolate if continues num of NAN is within threshold
        df[col].interpolate(method='spline', limit=threshold,axis=0)
        return df

    def handle_limit(self,data,col,lower,upper):
        df=deepcopy(data)
        #### give NAN if over the lower/upper limit
        df[col] = df[col].apply(lambda x:np.nan if (x>upper and x<lower) else x)
        return df

    def handle_constant(self,data,col,capacity,threshold):
        df=deepcopy(data)
        #### continues constant array's std must be zero
        df['constant_std'] = df[col].rolling(window=threshold).std()
        df['constant_mu'] = df[col].rolling(window=threshold).mean()
        df['index']=range(len(df)) ##data's index are date so....
        condition=df['constant_std']==0 and (df['constant_mu']!=0 or df['constant_mu']!=capacity)
        constant_end_index=df[condition]['index'] ##find end index
        constant_index=[] ##find all index not wanted
        for i in constant_end_index:
            constant_index+=[j for j in range(i-threshold+1,i+1)]
        df['index']=df['index'].apply(lambda x: False if x in constant_index else True)
        df=df[df['index']==True] ##remain index wanted
        del df['index'],df['constant_std'],df['constant_mu'] ##delete auxiliary varible
        return df

    def download_suntime(self,area,belong,longitude,latitude,begin,end):
        """
        根据输入的经度、维度，下载从起始到结束之间的每天日出日落时间
        该函数使用的时候需要联网环境
        Parameters
        ----------
        area : str
            目标区域名称.
        belong : str
            目标区域所属区域/国家名称.
        longitude : float
            经度.
        latitude : float
            纬度.
        begin : str
            起始时间 “2020-1-1”
        end : str
            结束时间 “2023-1-1”.
    
        Returns
        -------
        None.
    
        """
    
        #example:    
        # city = LocationInfo("Nanjing", "China", "China/Nanjing", latitude, longitude)#纬度，经度
        city = self.LocationInfo(area, belong, f"{belong}/{area}", latitude, longitude)#纬度，经度
        logging.info((
            f"Information for {city.name}/{city.region}\n"
            f"Timezone: {city.timezone}\n"
            f"Latitude: {city.latitude:.02f}; Longitude: {city.longitude:.02f}\n"
        ))
        
        #将输入的起始和结束时间转换为指定的date格式
        begin_list = begin.split("-")
        end_list = end.split("-")
        begin = date(int(begin_list[0]),int(begin_list[1]),int(begin_list[2]))
        end = date(int(end_list[0]),int(end_list[1]),int(end_list[2]))
    
        #根据起始和结束时间，遍历的获取每日的日出、日落时间    
        date_list = []
        sunrise_list = []
        sunset_list = []
        for i in range((end - begin).days + 1):
            current_date = begin + timedelta(days=i)
            current_date_str = current_date.strftime('%Y-%m-%d')
            s = self.sun(city.observer, date=current_date)
            
            #modify UTC timezone into BeiJing timezone
            sunrise_list.append(s["sunrise"]+timedelta(hours=8))
            sunset_list.append(s["sunset"]+timedelta(hours=8))
            date_list.append(current_date_str)
            
        data = {"Date":date_list,"Sunrise":sunrise_list,"Sunset":sunset_list}
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] =  df["Date"].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
        df["Sunrise"] =  df["Sunrise"].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
        df["Sunset"] =  df["Sunset"].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
    
        return df    


    def sunrise_before(self,x):
        """
        找到日出前的一个15分钟间隔的点
        """
        x = pd.to_datetime(x)
        #取分钟数据，并转换成int
        time = x.strftime("%Y-%m-%d %H")
        M = int(x.strftime("%M"))
        if M<=15:
            M=0
        elif M<=30:
            M=15
        elif M<=45:
            M=30
        elif M<=60:
            M=45
        time = time+f":{M}"
        return time

    def sunset_after(self,x):
        """
        找到日落后的第一个15分钟间隔的点
        """
        x = pd.to_datetime(x)
        #取分钟数据，并转换成int
        time = x.strftime("%Y-%m-%d ")
        M = int(x.strftime("%M"))
        H = int(x.strftime("%H"))
        if M<=15:
            M=15
        elif M<=30:
            M=45
        elif M<=45:
            M=0
            H = H+1
        elif M<=60:
            M=0
            H = H+1
        time = time+f"{H}:{M}"
        return time
            
    def sunset_zero(self,data,cols):
        """
        函数功能：将日落时间段， 
        辐照度和负荷列["GlobalR","DirectR","DiffuseR","load"]
        强制设置为0
        """
        
        #将suntime数据中指定的时间信息列，转换成时间类型
        sun_date_cols = ["Sunrise","Sunset","sunrise_before","sunset_after"]
        for col in sun_date_cols:
            self.suntime[col] = pd.to_datetime(self.suntime[col])
        
        df = copy.deepcopy(data)
        #读取原始数据的列名，用于返回函数处理结果时返回一模一样的列
        ori_col = df.columns.tolist()
        ori_col.append("sunset_flag")
        
        #为df创建一天为单位的时间列dayofdate
        df["dayofdate"] = df["date"].apply(lambda x:x.strftime("%Y-%m-%d"))
        
        #处理日出前的置零工作
        sunrise_time = self.suntime.loc[:,["sunrise_before","start_zero_flag"]]
        df = pd.merge(left =df,right= sunrise_time,left_on="date",right_on="sunrise_before",how="left")
        df = df.reset_index(drop=True)
    
        #按天进行分组，然后进行start_zero_flag向后填充
        def _bfill_flag(x):
            #每天日出前的flag填充为0
            x["start_zero_flag"].fillna(method="bfill",inplace=True)
            #将剩下的点填充为1
            x["start_zero_flag"].fillna(1,inplace=True)
            return x
        grouped = df.groupby("dayofdate")
        df = grouped.apply(_bfill_flag)
    
        #筛选出日落的index，将相应的属性列在置零
        sunset_index = df[df["start_zero_flag"]==0].index.tolist()
        for col in cols:
            df.loc[sunset_index,col] = 0
        #创建一列sunset_flag，来标记日落的时间段。日落=1，其他=0
        df["sunset_flag"] = 0
        df.loc[sunset_index,"sunset_flag"] = 1
        
        #选择df原有的列，接着继续进行日落后数据清零的工作
        df = df.loc[:,ori_col]
        
        #为df创建一天为单位的时间列dayofdate
        df["dayofdate"] = df["date"].apply(lambda x:x.strftime("%Y-%m-%d"))
        #处理日落后的置零工作
        sunset_time = self.suntime.loc[:,["sunset_after","start_zero_flag"]]
        df = pd.merge(left =df,right= sunset_time,left_on="date",right_on="sunset_after",how="left")
        df = df.reset_index(drop=True)
    
        #按天进行分组，然后进行start_zero_flag向后填充
        def _ffill_flag(x):
            #每天日出前的flag填充为0
            x["start_zero_flag"].fillna(method="ffill",inplace=True)
            #将剩下的点填充为1
            x["start_zero_flag"].fillna(1,inplace=True)
            return x
        grouped = df.groupby("dayofdate")
        df = grouped.apply(_ffill_flag)
    
        #筛选出日落的index，将相应的属性列在置零
        sunset_index = df[df["start_zero_flag"]==0].index.tolist()
        for col in cols:
            df.loc[sunset_index,col] = 0
        #标记日落的时间段。日落=1，其他=0
        df.loc[sunset_index,"sunset_flag"] = 1
        
        #选择df原有的列，并返回处理后的数据
        df = df.loc[:,ori_col]
        
        return df

