# -*- coding: utf-8 -*-
"""
@author: cty
"""

import os
import pandas as pd
import glob
import json
from copy import deepcopy

# =============================================================================

lgb_config = {
    'boosting_type': 'gbdt',
    'class_weight': None,
    'colsample_bytree': 1.0,
    'device': 'cpu',
    'importance_type': 'split',
    'learning_rate': 0.044,
    'max_depth': 10,
    'min_child_samples': 91,
    'min_child_weight': 0.001,
    'min_split_gain': 0.2,
    'n_estimators': 140,
    'n_jobs': -1,
    'num_leaves': 31,
    'objective': 'regression',
    'random_state': 123,
    'reg_alpha': 0.9,
    'reg_lambda': 0.6,
    'silent': True,
    'subsample': 0.4,
    'subsample_for_bin': 200000,
    'subsample_freq': 0
}
xgb_config = {
    'booster': 'gbtree',
    'n_estimators': 170,
    'max_depth': 3,
    'learning_rate': 0.0621,
    'subsample': 1,
    'gamma': 0.3,
    'reg_alpha': 0.9,
    'reg_lambda': 0.2,
    'nthread': 8,
    'objective': 'reg:squarederror',
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'gpu_id': -1,
    'tree_method': 'auto',
}
lr_config = {
    'learning_rate': 10e-3,
    'epoch': 10,
    'batch_size': 64,
    'num_input': 4,
    'num_hidden': 256,
    'num_output': 1,
    'early_stop_round': 10,
}
lstm_config = {
    'learning_rate':    0.0001,
    'epochs':           200, ##200
    'batch_size':       64,
    'seq_len':          96,
    'output_len':       96,
    'early_stop_round': 30,
}
knn_config = {
    'algorithm':    'auto',
    'leaf_size':    30,
    'metric':       'minkowski',
    'n_jobs':       -1,
    'n_neighbors':  50,
    'p':            2,
    "weights":      'uniform'
}
lasso_param = {
        'alpha':0.0,
        'copy_X':True,
        'fit_intercept':True,
        'max_iter':1000,
        #'normalize':True,
        'positive':False,
        'precompute':False,
        'random_state':5275,
        'selection':'cyclic',
        'tol':0.0001,
        'warm_start':False
        }
# =============================================================================
class EnvConfig:
    def __init__(self):
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        # ftp连接配置
        self.ftp_host = "120.76.75.9"
        self.ftp_port = 212
        self.ftp_username = "GansuNariUpload" 
        self.ftp_password = "sdhsih@67as"

        # # 数据库配置
        # self.database_url = 'jdbc:dm://172.28.238.5:5236/'
        # self.database_user = 'SYSDBA'
        # self.database_password = 'SYSDBA'
        # self.database_dirver = 'dm.jdbc.driver.DmDriver'
        # self.database_jarFile = '/opt/dm8/drivers/jdbc/DmJdbcDriver18.jar'
        # self.database_driver_odbc = '/opt/dm8/drivers/odbc/libdodbc.so'
        # self.database_ip = '172.28.238.5'

        # 数据库配置
        self.database_url = '153.3.1.201'
        self.database_user = 'xnyyc'
        self.database_password = 'xnyyc@123'
        self.database_dirver = 'dm.jdbc.driver.DmDriver'
        # TODO 修改成现场项目绝对路径地址
        #self.database_jarFile = '/home/nusp/jiaxing/load_forcast/newenergy_jiaxing/zj_jiaxing/dm_driver/DmJdbcDriver.jar'
        self.database_jarFile = '/home/nusp/jiaxing/load_forcast/newenergy_jiaxing/zj_jiaxing/dm_driver/dm.jdbc.driver.dm7-7.1.5.jar'
        self.database_driver_odbc = '/home/nusp/jiaxing/load_forcast/newenergy_jiaxing/zj_jiaxing/dm_driver/libdodbc.so'
        self.database_ip = '153.3.1.201'

class AreaConfig:
    def __init__(self, area):
        self.area = area
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.area_path = os.path.join(self.project_path, "data","area", area)
        self.area_info = pd.read_csv(os.path.join(self.project_path, 'data','area_info.csv'))
        
        self.capacity = self.area_info[self.area_info["FarmCode"] == area]['capacity'].iloc[0]
        self.Longitude = self.area_info[self.area_info["FarmCode"] == area]['Longitude'].iloc[0]
        self.Latitude = self.area_info[self.area_info["FarmCode"] == area]['Latitude'].iloc[0]
        self.area_type = self.area_info[self.area_info["FarmCode"] == area]['area_type'].iloc[0]

        self.models_used = self.area_info[self.area_info["FarmCode"] == area]['models_used'].iloc[0]
        self.feas_used = self.area_info[self.area_info["FarmCode"] == area]['feas_used'].iloc[0].split('+')
        
        #In json.loads(dict), '' for key and value must be replaced by "" in dict, use true not True 
        self.xgb_config = json.loads(self.area_info[self.area_info["FarmCode"] == area]['xgb_config'].iloc[0])
        self.lgb_config = json.loads(self.area_info[self.area_info["FarmCode"] == area]['lgb_config'].iloc[0])
        self.lr_config = json.loads(self.area_info[self.area_info["FarmCode"] == area]['lr_config'].iloc[0])
        
        self.day_point = self.area_info[self.area_info["FarmCode"] == area]['day_point'].iloc[0]   
        self.trend = self.area_info[self.area_info["FarmCode"] == area]['trend'].iloc[0].split('+')
        

if __name__ == '__main__':
    config=AreaConfig('NARI-19012-Xibei-gbdyfd')