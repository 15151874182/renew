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

from datetime import date, timedelta
from renew_center.tools.logger import setup_logger
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
              Longitude, ##None for wind clean
              Latitude): ##None for wind clean

#preparing ====================================================================        
        day_point=96  ##15min time freq everyday
        df=deepcopy(data)
        df=df[selected_feas]
        if area_type=='pv' and online==True:
            from astral.sun import sun
            from astral import LocationInfo     
            self.sun=sun
            self.LocationInfo=LocationInfo            
            #suntime是日出日落信息，需要联网
            self.suntime = self.download_suntime(area=area,
                                                belong='china',
                                                Longitude=Longitude,
                                                Latitude=Latitude,
                                                begin="2018-1-1",
                                                end="2024-12-31")
            self.suntime["sunrise_before"] = self.suntime["Sunrise"].apply(lambda x:pd.to_datetime(x).round('15T'))
            self.suntime["sunset_after"] = self.suntime["Sunset"].apply(lambda x:pd.to_datetime(x).round('15T'))  
            self.suntime.to_csv('data/suntime.csv')
            
        elif area_type=='pv' and online==False:
            self.suntime = pd.read_csv('data/suntime.csv')
            
# =============================================================================             
        for col in selected_feas: ##every feas should handle_NAN,handle_limit,handle_constant
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
            
#handle_NAN =================================================================  
            df=self.handle_NAN(df,col,threshold=day_point//6)
#handle_limit================================================================            
            df=self.handle_limit(df,col,lower,upper)
#special strategy for wind/pv================================================================             
            if area_type=='wind':
                pass  ##special strategy for wind if have will be put here 
            elif area_type=='pv':
                if ("rad" or 'load') in col:  ## fea related to suntime
                    df=self.sunset_zero(df,col)
#handle_constant================================================================        
        ##constant=0 or capacity will be ignored,constant=0 problem will be put into similarity detect part             
            df=self.handle_constant(df,col,capacity,threshold=day_point//6) 
            xx=1
        
    def clean_area(self,config,selected_feas):
        return self.clean(data=config.data,
                          selected_feas=selected_feas,
                          use_type=config.use_type,  ##'train'/'predcit'
                          capacity=config.capacity,
                          Longitude=config.Longitude,
                          Latitude=config.Latitude)
    
    def check_length(self,data,day_point,threshold):
        days=len(data)//day_point
        ##return True if number of data is enough
        return True if days>=threshold else False

    def handle_NAN(self,data,col,threshold):
        df=deepcopy(data)
        #### spline interpolate if continues num of NAN is within threshold
        df[col].interpolate(method='cubic', limit=threshold,axis=0)
        return df

    def handle_limit(self,data,col,lower,upper):
        df=deepcopy(data)
        #### give NAN if over the lower/upper limit
        df[col] = df[col].apply(lambda x:np.nan if (x>upper and x<lower) else x)
        return df

    def handle_constant(self,data,col,capacity,threshold):
        '''
        delete threshold points constant data[col] except constant=0 or capacity
        threshold->int, how many continues points considered
        '''
        df=deepcopy(data)
        #### continues constant array's std must be zero
        df['constant_std'] = df[col].rolling(window=threshold).std()
        df['constant_mu'] = df[col].rolling(window=threshold).mean()
        df['index']=range(len(df)) ##data's index are date so....
        condition=(df['constant_std']==0) & ((df['constant_mu']!=0) | (df['constant_mu']!=capacity))
        constant_end_index=df[condition]['index'] ##find end index
        constant_index=[] ##find all index not wanted
        for i in constant_end_index:
            constant_index+=[j for j in range(i-threshold+1,i+1)]
        df['index']=df['index'].apply(lambda x: False if x in constant_index else True)
        df=df[df['index']==True] ##remain index wanted
        del df['index'],df['constant_std'],df['constant_mu'] ##delete auxiliary varible
        return df

    def download_suntime(self,area,belong,Longitude,Latitude,begin,end):
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
        city = self.LocationInfo(area, belong, f"{belong}/{area}", Latitude, Longitude)#纬度，经度
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

    def sunset_zero(self,data,col):
        df=deepcopy(data)
        sunset_point=pd.DataFrame(sorted(pd.concat([self.suntime['sunrise_before'],self.suntime['sunset_after']]).drop_duplicates()))
        sunset_point.columns=['sunset_point']
        sunset_point.index=pd.to_datetime(sunset_point['sunset_point']) ##change index into datetime
        df=df.join(sunset_point)
        sunset_list=list(df[~df['sunset_point'].isna()]['sunset_point']) ##sunset Separation point list
        res=[] ##full sunset point list
        for idx,time in enumerate(sunset_list):
            if idx==0 and time.hour<12: ##create first point sunset points 
                res+=list(pd.date_range(time-pd.Timedelta(8,unit='H'),time,freq='15T'))
            elif idx==len(sunset_list)-1 and time.hour>12: ##create last point sunset points 
                res+=list(pd.date_range(time,time+pd.Timedelta(8,unit='H'),freq='15T'))
            elif time.hour>12:  ##create middle sunset points 
                res+=list(pd.date_range(sunset_list[idx],sunset_list[idx+1],freq='15T'))
        df['date']=df.index
        df['sunset_flag']=df['date'].apply(lambda x:True if x in res else False)
        df[col][df['sunset_flag']==True]=0 ##sunset points set to zero
        del df['sunset_flag'],df['date'],df['sunset_point']
        return df