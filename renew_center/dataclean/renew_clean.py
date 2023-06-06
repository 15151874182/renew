

import os
import pandas as pd
import datetime
import numpy as np
from scipy.signal import argrelextrema
import time
import copy
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
from center.tools.common import fix_date
from center.dataclean.detect_tools import check_nan,check_limit,detect_constant,detect_similarity
from center.dataclean.fill_tools import interp_in_day

class Clean():
    def __init__(self,
                 station_type,
                 use_type,
                 station_name,
                 capacity,
                 freq,
                 longitude,
                 latitude,
                 similarity_detect,
                 threshold):

        self.station_type=station_type
        self.use_type=use_type
        self.station_name=station_name
        self.capacity=capacity
        self.freq=freq
        self.longitude=longitude
        self.latitude=latitude
        self.similarity_detect=similarity_detect
        self.threshold=threshold
        
        if self.station_type=='wind':
            print('using Wind_clean block.....')
            self.clean_station=self.clean_wind
        elif self.station_type=='pv':
            print('using Solar_clean block.....')
            try:
                from astral.sun import sun
                from astral import LocationInfo
                self.sun=sun
                self.LocationInfo=LocationInfo
            except:
                print('please pip install astral  -i https://pypi.tuna.tsinghua.edu.cn/simple')            
            self.clean_station=self.clean_solar
            
            #suntime是日出日落信息，需要联网
            self.suntime = self.download_suntime(area=self.station_name,
                                                belong='china',
                                                longitude=longitude,
                                                latitude=latitude,
                                                begin="2018-1-1",
                                                end="2023-12-31")
            self.suntime["sunrise_before"] = self.suntime["Sunrise"].apply(lambda x:self.sunrise_before(x))
            self.suntime["sunset_after"] = self.suntime["Sunset"].apply(lambda x:self.sunset_after(x))
            self.suntime["start_zero_flag"] = 0   
# =============================================================================
####光伏辅助函数
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
        print((
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
            # s = sun(area_GF.observer, date=datetime.date(2021,10,28))
            s = self.sun(city.observer, date=current_date)
            
            #打印信息例子，平时可以注释掉
            # print((
            #     f'Dawn:    {s["dawn"]+timedelta(hours=8)}\n'
            #     f'Sunrise: {s["sunrise"]+timedelta(hours=8)}\n'
            #     f'Noon:    {s["noon"]+timedelta(hours=8)}\n'
            #     f'Sunset:  {s["sunset"]+timedelta(hours=8)}\n'
            #     f'Dusk:    {s["dusk"]+timedelta(hours=8)}\n'
            # ))
            
            #默认返回时间是 UTC timezone，需要修改返回为北京时间
            sunrise_list.append(s["sunrise"]+timedelta(hours=8))
            sunset_list.append(s["sunset"]+timedelta(hours=8))
            date_list.append(current_date_str)
            
        #将数据整合到dataframe中
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
        def bfill_flag(x):
            #每天日出前的flag填充为0
            x["start_zero_flag"].fillna(method="bfill",inplace=True)
            #将剩下的点填充为1
            x["start_zero_flag"].fillna(1,inplace=True)
            return x
        grouped = df.groupby("dayofdate")
        df = grouped.apply(bfill_flag)
    
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
        def ffill_flag(x):
            #每天日出前的flag填充为0
            x["start_zero_flag"].fillna(method="ffill",inplace=True)
            #将剩下的点填充为1
            x["start_zero_flag"].fillna(1,inplace=True)
            return x
        grouped = df.groupby("dayofdate")
        df = grouped.apply(ffill_flag)
    
        #筛选出日落的index，将相应的属性列在置零
        sunset_index = df[df["start_zero_flag"]==0].index.tolist()
        for col in cols:
            df.loc[sunset_index,col] = 0
        #标记日落的时间段。日落=1，其他=0
        df.loc[sunset_index,"sunset_flag"] = 1
        
        #选择df原有的列，并返回处理后的数据
        df = df.loc[:,ori_col]
        
        return df
# =============================================================================

# =============================================================================
#### 风电辅助函数
    def wind_cls(self,df,time_col,wind_col):
        """
        对输入数据中的风速进行风过程的分类，返回带有标记的列 
            低风力：low_wind
            小波动：small_fluctuation
            持续波动：Continuous_fluctuation
            大波动：large_fluctuation
            双峰出力：double_power
        Parameters
        ----------
        df : DataFrame
            包含时间和风速列的数据.
        time_col : str
            时间列名称.
        wind_col : str
            需要进行分类的风速列名称.
    
        Returns
        -------
        df : DataFrame
            带有风过程标记的数据.
        """
        start = time.time()
        
    #     df = data.loc[:,[time_col,wind_col]].copy(deep=True)
        #对列重命名
    #     df.columns = ["time","wind"]
        
        #计算风速3次方列
        wind_3 = "wind**3"
    #     df[wind_3] = df["wind"]**3
        df[wind_3] = df[wind_col]**3
    
        #计算阈值T
        T_1 = 0.5 * sum(df[wind_3]) / len(df[wind_3])
        T_2 = 1.0 * sum(df[wind_3]) / len(df[wind_3])
        
        #类风能序列
        w = np.array(df[wind_3])
        #类风能序列局部极小值序列的索引
        w_min = list(argrelextrema(w,np.less)[0])
        #类风能序列局部极大值序列的索引
        w_max = list(argrelextrema(w,np.greater)[0])
        
        df["w_min"]=0
        df.loc[w_min,"w_min"] = 1
            
        df["w_max"]=0
        df.loc[w_max,"w_max"] = 1
            
        w_cap = max(df[wind_3])
        
        df["low_wind"]=0 
    #     df["small_fluctuation"]=0
    #     df["Continuous_fluctuation"]=0
    #     df["large_fluctuation"]=0
    #     df["double_power"]=0
    
        #低风力出力

        for f in range(0,len(w_min)-1):
            for i in range(f+1,len(w_min)):
            
                #w_1为w[w_min[i]]  w_n为w[w_min[i+bt]] 值 | w_min[i]和w_min[i+bt]是极小值的 索引
                if w[w_min[f]]>T_1:
                    break
                if w[w_min[i]]>T_1:
                    continue
                
                ls_a=[]
                
                for j in w_max: 
                    if j>w_min[f] and j<w_min[i]: #w_1和w_n之间包含的极大值点
                        ls_a.append(j)  #ls中存放介于w1和wn之间极大值的索引
                        if j>w_min[i]:
                            break
                
                if len(ls_a)!=0:
                
                    #l为w_1至w_n之间极大值点个数
                    # l_a = len(ls_a)
                           
                    #最近相邻w_max_-1
                    idx_left = w_max.index(ls_a[0])
                    if idx_left != 0:
                        w_max_1 = w[w_max[idx_left-1]]
                    else:
                        break
                        
                    #最近相邻w_max_l+1
                    idx_right = w_max.index(ls_a[-1])
                    if idx_right != len(w_max)-1:
                        w_max_l1 = w[w_max[idx_right+1]] 
                    else:
                        break
                    
                    if w_max_1<=T_1 or w_max_1>w_cap:
                        break            
                    if w_max_l1<=T_1 or w_max_l1>w_cap:
                        continue
                            
                    #根据极大值点索引找到每个值和T比较
                    if (all(w[k] <= T_1 for k in ls_a)):
                        df.loc[w_min[f]:w_min[i]-1,"low_wind"]=1
                    else:
                        break
                
                    # print("低风力--极小值间极大值的个数为：",l_a)
                else:
                    # print("低风力--极大值列表为空!")
                    continue
            
        print("low_wind detect finished!")    
        return df        
# =============================================================================

# =============================================================================
#### 通用辅助函数

    def get_del_list(self,data,time_col,col,day_len):
        """
        按照以下逻辑，将需要删除的数据日期列表返回
        #1、一天中存在异常的值占了50%就是删除
        #2、一天中存在连续缺失3个小时的就删除
        
        Parameters
        ----------
        df : Dataframe
            总数据集.
        col : str
            需要处理的列名
        day_len : int 
            按天划分时，一天的数据长度
        """
        df = copy.deepcopy(data)
        #创建以天为单位的列dayofdate
        df[time_col] = pd.to_datetime(df[time_col])
        df["dayofdate"] = df[time_col].apply(lambda x:x.strftime("%Y-%m-%d"))
        #用于保存需要删除的天的列表
        del_day=[]
        def ab_day(x):
            #筛选存在异常的数据
            temp = x[x["abnormal_flag"]!=0]
            
            #超过40%异常点的日期保存下来
            if len(temp)>0.4*day_len:
                del_day.append(temp["dayofdate"].iloc[0])
    
            #连续缺失4个小时的日期保存下来
            elif 4 in temp["abnormal_flag"].tolist():
                del_day.append(temp["dayofdate"].iloc[0])
            return x
        
        #使用group对数据按天进行分组
        grouped = df.groupby("dayofdate")
        df = grouped.apply(ab_day)
        
        return del_day    
    
    def del_by_date(self,data,date_list,time_col="date",day_end="23:45"):
        """
        根据要删除的时间列表，按天在原数据上进行删除
        df: 原始数据dataframe
        date_list: 需要删除的日期列表，以天为单位。
        """
        df = data.copy()
        for i,temp_day in enumerate(date_list):
            if isinstance(temp_day, str):
                pass;
            else:
                #将天的日期转换成字符串形式
                temp_day = temp_day.strftime('%Y-%m-%d')
    #        print(f"del {i} day: {temp_day}")
            
            start_time = "{} 00:00".format(temp_day)
            end_time = "{} {}".format(temp_day,day_end)
            
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            
            #找到原始数据中相应数据的index
            temp = df[np.logical_and(df[time_col]>=start_time,df[time_col]<=end_time)].index.tolist()
            
            #在数据集上删除相应的index
            df = df.drop(index=temp,axis=0)
        print("del {}".format(len(date_list)))
        df = df.reset_index(drop=True)
         
        return df    
    
# =============================================================================


# =============================================================================
####主接口

# =============================================================================

    # def clean_solar_process(self,
    #                         df,
    #                         col,
    #                         time_col,
    #                         upper,
    #                         lower,
    #                         sunset_cols):
    #     """
    #     清洗的流程
    #     Parameters
    #     ----------
    #     df : Dataframe
    #         需要处理的数据.
    #     col : str
    #         需要清洗的列名.
    #     time_col : str
    #         时间戳列名.
    #     tmp_col : str
    #         温度列名称.
    #     cap : int
    #         数值的上限.
    #     lower : int
    #         数值的下限.
    #     suntime : Dataframe
    #         包含日出日落时间段的表格.
    #     sunset_cols : list
    #         需要经过日落时段置零的列名.
    #         ["GlobalR","DirectR","DiffuseR","load"]
    #     Returns
    #     -------
    #     df.
    
    #     """
        
    #     df[time_col] = pd.to_datetime(df[time_col])
    #     if col in sunset_cols:
    #         print("enter sunset period detect!")
    #         #先将日落时间段相应的列全部置0，并且新增一列sunset_flag，为1时表示为日落时间
    #         df = self.sunset_zero(df,cols=[col])
    #         #将数据分割为日落和非日落时段，日落时段数据不参与下面的数据清洗流程
    #         sunset_part = df[df["sunset_flag"]==1].copy(deep=True)
    #         sunset_part.drop(columns=["sunset_flag"],inplace=True)
    #         sunset_part = sunset_part.reset_index(drop=True)
    #         df = df[df["sunset_flag"]!=1].copy(deep=True)
    #         df.drop(columns=["sunset_flag"],inplace=True)
    #         df = df.reset_index(drop=True)
        
    #     # 进行异常检测，nan值、极大极小异常值、连续恒定值
    #     df["abnormal_flag"] = 0
    #     #检测nan值标记为 abnormal_flag=1
    #     df.loc[check_nan(df,col = col),"abnormal_flag"] = 1        
    #     #进行上下限检测,标记为abnormal_flag = 2
    #     df.loc[check_limit(df, col=col, upper=upper, lower=lower),"abnormal_flag"] = 2
        
    #     #检测连续恒定值，检测缺失大于1个小时的数据，并标记为abnormal_flag = 3
    #     df.loc[detect_constant(df,col,n=int(self.freq/24),limit=0.001,cap=self.capacity),"abnormal_flag"] = 3 
    #     #检测连续恒定值，检测缺失大于3个小时的数据（后期直接按天删除），并标记为abnormal_flag = 4
    #     df.loc[detect_constant(df,col,n=int(self.freq/8),limit=0.001,cap=self.capacity),"abnormal_flag"] = 4
                    
    #     #将异常的值全部置为nan
    #     df.loc[df[df["abnormal_flag"]!=0].index.tolist(),col] = np.nan
        
    #     #按照逻辑，获取需要按天删除的日期列表,对于辐照度等数据按天删除的逻辑要更加严格，因为有一半数据黑夜
    #     if col in sunset_cols:
    #         del_day = self.get_del_list(df, time_col=time_col, col=col, day_len=self.freq*0.5)
    #     else:
    #         del_day = self.get_del_list(df, time_col=time_col, col=col, day_len=self.freq)
        
    #     if self.freq==96:
    #         day_end='23:45'
    #     elif self.freq==288:
    #         day_end='23:55'
        

    #     # #这里需要拼接回sunset_part
    #     # if col in sunset_cols:
    #     #     df = pd.concat([df,sunset_part],axis=0)
    #     #     df = df.sort_values(time_col, ascending=True)
    #     #     df = df.reset_index(drop=True)

    #     if self.use_type=='train' and col in sunset_cols:
    #         #训练时候可以删除脏数据，按天删除
    #         df = self.del_by_date(df, date_list=del_day, time_col=time_col, day_end=day_end)
    #         sunset_part = self.del_by_date(sunset_part, date_list=del_day, time_col=time_col, day_end=day_end)
    #     elif self.use_type=='train' and col not in sunset_cols:
    #         df = self.del_by_date(df, date_list=del_day, time_col=time_col, day_end=day_end)
    #     elif self.use_type=='test' and col in sunset_cols:
    #         #测试时候数据再脏也只能修补
    #         df = interp_in_day(df,col,time_col)
    #         sunset_part = interp_in_day(sunset_part,col,time_col) 
    #     elif self.use_type=='test' and col not in sunset_cols:
    #         df = interp_in_day(df,col,time_col)
                
    #     if col in sunset_cols:
    #         #将清洗后的df和sunset_part进行拼接
    #         #为sunset_part初始化abnormal_flag=0
    #         sunset_part["abnormal_flag"]=0
    #         df = pd.concat([df,sunset_part],axis=0)
    #         df = df.sort_values(time_col, ascending=True)
            
    #     df = df.reset_index(drop=True)
        
    #     #删除abnormal_flag列
    #     df = df.drop(columns="abnormal_flag")
    #     df = df.reset_index(drop=True)
        
    #     return df, del_day
    def clean_solar_process(self,
                            df,
                            col,
                            time_col,
                            upper,
                            lower,
                            sunset_cols):
        """
        清洗的流程
        Parameters
        ----------
        df : Dataframe
            需要处理的数据.
        col : str
            需要清洗的列名.
        time_col : str
            时间戳列名.
        tmp_col : str
            温度列名称.
        cap : int
            数值的上限.
        lower : int
            数值的下限.
        suntime : Dataframe
            包含日出日落时间段的表格.
        sunset_cols : list
            需要经过日落时段置零的列名.
            ["GlobalR","DirectR","DiffuseR","load"]
        Returns
        -------
        df.
    
        """
        #通用变量和流程定义
        df[time_col] = pd.to_datetime(df[time_col])
        if self.freq==96:
            day_end='23:45'
        elif self.freq==288:
            day_end='23:55'
        def __process(df): ##为了减少重复代码的内部函数，不对外调用
            # 进行异常检测，nan值、极大极小异常值、连续恒定值
            df["abnormal_flag"] = 0
            #检测nan值标记为 abnormal_flag=1
            df.loc[check_nan(df,col = col),"abnormal_flag"] = 1        
            #进行上下限检测,标记为abnormal_flag = 2
            df.loc[check_limit(df, col=col, upper=upper, lower=lower),"abnormal_flag"] = 2
            #检测连续恒定值，检测缺失大于1个小时的数据，并标记为abnormal_flag = 3
            df.loc[detect_constant(df,col,n=int(self.freq/24),limit=0.001,cap=self.capacity),"abnormal_flag"] = 3 
            #检测连续恒定值，检测缺失大于3个小时的数据（后期直接按天删除），并标记为abnormal_flag = 4
            df.loc[detect_constant(df,col,n=int(self.freq/8),limit=0.001,cap=self.capacity),"abnormal_flag"] = 4
            #将异常的值全部置为nan
            df.loc[df[df["abnormal_flag"]!=0].index.tolist(),col] = np.nan   
            ##获取要删除天数list
            del_day = self.get_del_list(df, time_col=time_col, col=col, day_len=self.freq)                
            return df,del_day
        
        ##跟太阳有关的特征都要先拆分白天部分进行清洗，找出del_list，然后拼回黑夜,trian模式就按天删除，test模式就插值
        ##跟太阳无关的特征不需要拆分，直接找出del_list, trian模式就按天删除，test模式就插值
        if col in sunset_cols: ##跟太阳有关的特征
            print("enter sunset period detect!")
            #先将日落时间段相应的列全部置0，并且新增一列sunset_flag，为1时表示为日落时间
            df = self.sunset_zero(df,cols=[col])
            #将数据分割为日落和非日落时段，日落时段数据不参与下面的数据清洗流程
            sunset_part = df[df["sunset_flag"]==1].copy(deep=True)
            sunset_part.drop(columns=["sunset_flag"],inplace=True)
            sunset_part = sunset_part.reset_index(drop=True)
            df = df[df["sunset_flag"]!=1].copy(deep=True)
            df.drop(columns=["sunset_flag"],inplace=True)
            df = df.reset_index(drop=True)  ##此处是拆分后的df            
            #清洗，并获取需要删除的天数list，但暂时还不改变df
            df,del_day=__process(df)        
            #将清洗后的df和sunset_part进行拼接,为sunset_part初始化abnormal_flag=0
            sunset_part["abnormal_flag"]=0
            df = pd.concat([df,sunset_part],axis=0)
            df = df.sort_values(time_col, ascending=True) ##此处是拼回后完整的df                                      
        else: #跟太阳无关的特征
            #清洗，并获取需要删除的天数list，但暂时还不改变df
            df,del_day=__process(df)                 

        #测试时候数据再脏也只能修补，不能删除
        if self.use_type=='test': 
            df = interp_in_day(df,col,time_col)
            del_day=[]
        #训练的时候按天删除
        elif self.use_type=='train': 
            ##整体按天删除
            df = self.del_by_date(df, date_list=del_day, time_col=time_col, day_end=day_end)      

        #删除abnormal_flag列
        df = df.drop(columns="abnormal_flag")
        df = df.reset_index(drop=True)
        
        return df, del_day
    def clean_solar(self,
                   df,
                   clean_col_list,
                   time_col,
                   load_col):
        self.total_days=int(len(df)/self.freq)
        # 求输入数据列名和选择列名的交集
        df_cols = df.columns.tolist()
        clean_col_list = list(set(clean_col_list) & set(df_cols))
        choose_list = clean_col_list + ["date"]
        
        print(f"choose_list :{choose_list}")
        df = df.loc[:, choose_list]
        
        #将cols列全部转换成数值类型
        for col in clean_col_list:
            df[col] = df[col].apply(pd.to_numeric, errors='coerce')
        #用于保存每种数据被删除的天数
        del_info = []
        del_day1=[]
        #循环依次清洗所有的列
        for i,clean_col in enumerate(clean_col_list):
            print(i,clean_col)
            #根据不同的列名来确定cap
            #全辐射
            if "Radiation" in clean_col:    
                upper = 1400
                lower = 0       
            #气温
            elif "temp" in clean_col:    
                upper = 48
                lower = -15
            #湿度
            elif "hum" in clean_col:    
                upper = 100
                lower = 0
            #气压
            elif "press" in clean_col:    
                upper = 1500
                lower = 640
            #功率
            elif "load" in clean_col:
                upper = self.capacity
                lower = 0
            else:
                upper=float('inf')
                lower=float('-inf')    
            print(f"正在处理{clean_col},upper={upper},lower={lower}")
            #需要指明温度列名称，以及收到日出日落影响的列名列表
            sunset_cols=[i for i in clean_col_list if ('load' in i or 'Radiation' in i)]
            df, del_day_sub = self.clean_solar_process(df,
                                                   col=clean_col,
                                                   time_col=time_col,
                                                   upper=upper,
                                                   lower=lower,
                                                   sunset_cols=sunset_cols)
            
            del_info.append(len(del_day_sub))
            del_day1+=del_day_sub
        
        del_day2=[]
        if self.similarity_detect==True:
            if self.use_type=='train':
                for fea in clean_col_list:
                    if 'Radiation' in fea:
                        df,del_day2=detect_similarity(df,col1='R_load',col2=fea,freq=self.freq,threshold=self.threshold) ##如果特征中存在风速列，选取第一个风速特征和load做相似度清洗
                        break
        #将删除信息变换成dataframe
        del_info = pd.DataFrame([clean_col_list,del_info])
        
        #设置第一行为columns名称
        del_info.columns = del_info.iloc[0]
        del_info = del_info[1:]
        del_info["trend"] =len(del_day2)
        del_info["reserve_days"] = len(df)/self.freq
        del_info["total_days"] =self.total_days
        del_info["station"] = self.station_name
        del_info['del_day']=' ;'.join(sorted(del_day1+del_day2))
        
        return df,del_info
    
    def clean_wind_process(self,
                           df,
                           col,
                           time_col,
                           load_col,
                           upper,
                           lower):
        """
        清洗的流程
        Parameters
        ----------
        df : Dataframe
            需要处理的数据.
        col : str
            需要清洗的列名.
        time_col : str
            时间戳列名.
        load_col : str
            load列名.可能叫load，power之类的
        cap : int
            DESCRIPTION.
        lower : int
            DESCRIPTION.
    
        Returns
        -------
        df.
    
        """
        df["abnormal_flag"] = 0
        
        # 进行异常检测，nan值、极大极小异常值、连续恒定值

        #检测nan值标记为 abnormal_flag=1
        df.loc[check_nan(df,col = col),"abnormal_flag"] = 1        
        #进行上下限检测,标记为abnormal_flag = 2
        df.loc[check_limit(df, col=col, upper=upper, lower=lower),"abnormal_flag"] = 2
        
        #检测连续恒定值，检测缺失大于1个小时的数据，并标记为abnormal_flag = 3
        df.loc[detect_constant(df,col,n=int(self.freq/24),limit=0.001,cap=self.capacity),"abnormal_flag"] = 3 
        #检测连续恒定值，检测缺失大于3个小时的数据（后期直接按天删除），并标记为abnormal_flag = 4
        df.loc[detect_constant(df,col,n=int(self.freq/8),limit=0.001,cap=self.capacity),"abnormal_flag"] = 4
        
        #在处理load之前，判断低风力非异常，防止删除低风力的情况
        if col==load_col and 'low_wind' in df.columns:
            df[df[load_col].notnull()]['abnormal_flag'][df['low_wind']==1]=0
        
        #将异常的值全部置为nan
        df.loc[df[df["abnormal_flag"]!=0].index.tolist(),col] = np.nan
        #按照逻辑，获取需要按天删除的日期列表
        del_day = self.get_del_list(df,time_col=time_col,col=col,day_len=self.freq)
        
        if self.freq==96:
            day_end='23:45'
        elif self.freq==288:
            day_end='23:55'

        if self.use_type=='train':
            #训练时候可以删除脏数据，按天删除
            df = self.del_by_date(df,date_list=del_day,time_col=time_col,day_end=day_end)      
            df = interp_in_day(df,col,time_col)
            if "speed" in col:    
                #低风力分类
                df = self.wind_cls(df, time_col=time_col, wind_col=col)
                df = df.drop(columns=['w_min','w_max','wind**3'])
                df = df.reset_index(drop=True)                
        elif self.use_type=='test':
            #测试时候数据再脏也只能修补
            df = interp_in_day(df,col,time_col)

        #删除abnormal_flag列
        df = df.drop(columns="abnormal_flag")
        df = df.reset_index(drop=True)
        
        return df,del_day


    def clean_wind(self,
                   df,
                   clean_col_list,
                   time_col,
                   load_col):
        '''
        对输入数据进行清洗
        df必须包含date列，load列，capacity列(一列值都为1个数)，特征列
        Parameters
        ----------
        df : Dataframe
            需要处理的数据.
        station : str
            场站名.
        clean_col_list:list
            需要清洗的列名
        time_col : str
            时间列名.
        load_col : str
            负荷列名.        
        Returns
        -------
        df del_info.
    
        '''
        self.total_days=int(len(df)/self.freq)
        #load放置在最后处理
        if self.use_type=='train':
            load_col = clean_col_list.pop(clean_col_list.index(load_col))
            clean_col_list.append(load_col)
    
        #截取需要的列
        choose_list = clean_col_list + [time_col]
        df = df.loc[:,choose_list]
        # #截取15分钟一个点
        # df = df[::3].reset_index(drop=True)
        
        #用于保存每种数据被删除的天数
        del_info = []
        del_day1=[]
        #循环依次清洗所有的列
        for clean_col in clean_col_list:
            #根据不同的列名来确定cap
            #风速
            if "speed" in clean_col:    
                upper = 35
                lower = 0
            #风向
            elif "dir" in clean_col:    
                upper = 360
                lower = 0
            #气温
            elif "temp" in clean_col:    
                upper = 48
                lower = -15
            #湿度
            elif "hum" in clean_col:    
                upper = 100
                lower = 0
            #气压
            elif "press" in clean_col:    
                upper = 1500
                lower = 640
            #功率
            elif "load" in clean_col:
                upper = self.capacity
                lower = 0
            else:
                upper=float('inf')
                lower=float('-inf')
            print(f"{clean_col},upper={upper},lower={lower}")
            
            if len(df)==0: ##数据小，前几个列的清洗就把数据删光了的情况
                print(f'data of {self.station_name} is bad!!!!!!!!!!')
                break
            
            df,del_day_sub=self.clean_wind_process(df=df,
                                     col=clean_col,
                                     time_col=time_col,
                                     load_col=load_col,
                                     upper=upper,
                                     lower=lower)
            
            del_info.append(len(del_day_sub))
            del_day1+=del_day_sub
        
        del_day2=[]
        if self.similarity_detect==True:
            if self.use_type=='train' and len(df)!=0:
                for fea in clean_col_list:
                    if 'speed' in fea:
                        df,del_day2=detect_similarity(df,col1='R_load',col2=fea,freq=self.freq,threshold=self.threshold) ##如果特征中存在风速列，选取第一个风速特征和load做相似度清洗
                        break

        #将删除信息变换成dataframe
        del_info = pd.DataFrame([clean_col_list,del_info])
        
        #设置第一行为columns名称
        del_info.columns = del_info.iloc[0]
        del_info = del_info[1:]
        del_info["trend"] =len(del_day2)
        del_info["reserve_days"] = len(df)/self.freq
        del_info["total_days"] =self.total_days        
        del_info["station"] = self.station_name
        del_info['del_day']=' ;'.join(sorted(del_day1+del_day2))
        if self.use_type=='train':
            try: del df['low_wind']
            except: pass
        
        return df,del_info
