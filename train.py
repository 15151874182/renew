import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import warnings
warnings.filterwarnings('ignore') 

config_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.abspath(os.path.dirname(config_path))

if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"add project_path:{project_path} to sys path")

import lightgbm as lgb
from zj_jiaxing.dataclean import get_cleaner, solve_pv_frog_leg
from center.model.linear_model import  StackingLr
from center.model.StackingModel import StakingModel
from center.model.xgb_model import XGBModel
from center.model.lgb_model import LGBModel
from center.model.lasso_model import LassoModel
from center.model.lstm_model import LSTMModel_stacking, LSTMModel_keras
from center.model.cnn_model import CnnModel
from center.model.knn_model import KNNModel

from config import config_parser
from feature import data_generator
from center.tools.accuracy import acc_mape, acc_mse, solar_wind_mse, MAPE
from center.tools.logger_function import get_logger
from center.dataclean.newEnergy_DataClean import Clean
from sklearn.model_selection import train_test_split

logger = get_logger("stacking")

def daily_mse_mape(df, C=None):
    import pandas as pd
    """
    按照天计算每天的mape和mase
    """
    df.datetime = pd.to_datetime(df.datetime)
    start_time = df.loc[df.index[0], "datetime"]
    end_time = df.loc[df.index[-1], "datetime"]
    target_time = start_time

    output = []
    while target_time < end_time:
        try:
            target_load = df.loc[df['datetime'] < target_time + pd.Timedelta(days=1)]

            target_load = target_load.loc[target_load['datetime'] >= target_time]
            target_load = target_load.reset_index(drop=True)
            _, rmse_nanwang, target_time_mse = solar_wind_mse(target_load.actual,
                                          target_load.predict_load, C)
            if np.isnan(target_time_mse):
                target_time = target_time + pd.Timedelta(days=1)
                continue
            target_time_mape = acc_mape(target_load.actual, target_load.predict_load, C, 0)
            print('date:', target_time.strftime("%Y-%m-%d"), 'rmse:',
                  target_time_mse, 'mape:', target_time_mape, "rmse_nanwang:", rmse_nanwang)
            temp = [target_time, target_time_mse, target_time_mape]
            output.append(temp)
            target_time = target_time + pd.Timedelta(days=1)
        except:
            target_time = target_time + pd.Timedelta(days=1)

    output = pd.DataFrame(output)
    output.columns = ["date", "mse", "mape"]

    return output

def test_eval(output, cap, daily_acc):
    _, _, _mse = solar_wind_mse(output['actual'], output['predict_load'], cap)
    _mape = acc_mape(output['actual'], output['predict_load'],cap,0)

    if daily_acc:
        try:
            print('\n')
            daily_mse_mape(output, cap)
        except:
            print("daily_mse_mape failed !")
    return _mse, _mape

class MyLGB(LGBModel):

    def set_params(self):
        # import platform
        para = self.config.get_para("lgb_param")
        # if platform.system().lower() == 'windows':
        # para['device']="gpu",
        # para["gpu_device_id"] = 0
        return para

    def build_model(self):
        lgb_param = self.set_params()

        model = lgb.LGBMRegressor(**lgb_param)
        return model

    def __train__(self, X_train, X_val, y_train, y_val):

        model = self.build_model()

        model.fit(X_train,
                  y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20,
                  eval_metric='rmse')

        return model

    def eval(self, output, daily_acc=False):
        cap = self.config.get_para("capacity")
        return test_eval(output, cap, daily_acc)

    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        predict_data = test_data.drop(columns=[self.load_name])
        predict_result = self.predict(predict_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final mape:{_mape}")
        return predict_result, _mse, _mape

class MyKnn(KNNModel):
    def eval(self, output, daily_acc=False):
        return test_eval(output, self.cap, daily_acc)

    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        predict_data = test_data.drop(columns=[self.load_name])
        predict_result = self.predict(predict_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result, _mse, _mape

class singleLSTM(LSTMModel_keras):
    def eval(self, output, daily_acc=False):
        cap = self.config.get_para("capacity")
        return test_eval(output, cap, daily_acc)

    def test(self, test_data, model_file=None):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        test_data[self.time_name] = pd.to_datetime(test_data[self.time_name])
        X_test, test_date = self.data_format(test_data, False)
        model_file = self.model_file if model_file is None else model_file
        model = self.load_model(model_file)
        y_pred = self.predict_model(model, X_test)
        output = pd.DataFrame()
        output['predict_load'] = y_pred
        output['datetime'] = test_date.reshape(-1)
        output['datetime'] = pd.to_datetime(output['datetime'])
        predict_result = pd.merge(left=output, right=test_data[[self.time_name, self.load_name]],
                          left_on="datetime", right_on=self.time_name, how="left")
        predict_result.rename(columns={self.load_name: "actual"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result, _mse, _mape

class MyLSTM(LSTMModel_stacking):
    def eval(self, output, daily_acc=False):
        cap = self.config.get_para("capacity")
        return test_eval(output, cap, daily_acc)

    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        # predict_data = test_data.drop(columns=[self.load_name])
        predict_data=test_data
        predict_result = self.predict(predict_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result, _mse, _mape

class MyCNN(CnnModel):
    def build_model(self):
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Model, Sequential, load_model
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
            from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, \
                Reshape
        except:
            import keras
            from keras.models import Model, Sequential, load_model
            from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
            from keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Reshape

        day_point = self.config.get_para("day_point") if "day_point" in self.config.config_info.keys() else 96
        feature_list = [n for n in self.config.get_para("feature_list") if n not in [self.load_name, self.time_name]]
        feature_num = len(feature_list)

        model = Sequential()
        model.add(
            Conv2D(64, (5, feature_num),
                   strides=(2, 1),
                   padding="same",
                   activation='relu',
                   input_shape=(day_point, feature_num, 1),
                   kernel_initializer=keras.initializers.RandomUniform(
                       -1e-3, 1e-3)))  # （时间步长，特征数，维度）

        # 卷积核数目 卷积核size 步长 补0策略 激活函数 输入格式    （24，16，32）
        # model.add(MaxPooling2D(pool_size=(2,2))) #下采样，输入长宽缩小两倍  （48，8，32）
        # model.add(Dropout(0.25))

        model.add(
            Conv2D(128, (5, 1),
                   strides=(2, 1),
                   padding="same",
                   activation='relu',
                   kernel_initializer=keras.initializers.RandomUniform(
                       -1e-3, 1e-3)))  # 卷积，输出通道数64 （48，8，64）

        #     model.add(MaxPooling2D(pool_size=(2,2)))  #下采样，输入长宽缩小两倍  （24，4，64）
        #     model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(
            Dense(units=256,
                  name="Dense",
                  activation='relu',
                  kernel_initializer=keras.initializers.RandomUniform(-1e-3,
                                                                      1e-3)))
        model.add(
            Dense(units=day_point,
                  name="Dense_2",
                  activation='relu',
                  kernel_initializer=keras.initializers.RandomUniform(-1e-3,
                                                                      1e-3)))

        adamax = keras.optimizers.Adam(lr=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08)

        model.compile(loss="mean_absolute_error", optimizer=adamax)
        return model

    def eval(self, output, daily_acc=False):
        return test_eval(output, self.cap, daily_acc)
    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        predict_data = test_data.drop(columns=[self.load_name])
        predict_result = self.predict(predict_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result, _mse, _mape

class MyLasso(LassoModel):
    def eval(self, output, daily_acc=False):
        return test_eval(output, self.cap, daily_acc)
    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        predict_data = test_data.drop(columns=[self.load_name])
        predict_result = self.predict(predict_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result, _mse, _mape

class MyXGB(XGBModel):
    def build_model(self):
        import xgboost as xgb
        model = xgb.XGBRegressor(**self.config.get_para("xgb_param"))
        return model
    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        predict_data = test_data.drop(columns=[self.load_name])
        predict_result = self.predict(predict_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result, _mse, _mape

    def eval(self, output, daily_acc=False):
        return test_eval(output, self.cap, daily_acc)


class Mylinear(StackingLr):

    def predict(self, predict_data, model_file=None):
        '''
        预测函数
        :param predict_data:    输入的预测数据
        :param model_file:      预测的模型路径
        :return:
        '''

        test_date = predict_data.pop(self.time_name).values
        # test_date = str(test_date[0].date())

        if model_file is None:
            model_file = self.model_file

        model = self.load_model(model_file)
        pre_result = self.predict_model(model, predict_data.values)

        # 预测结果保存
        result_df = pd.DataFrame()
        result_df["predict_load"] = pre_result.reshape(-1)
        result_df["date"] = test_date
        # result_df["area_name"] = self.area_name
        result_df.to_csv(self.predict_file, index=False)

        return result_df

    def eval(self, output, daily_acc=False):
        return test_eval(output, self.cap, daily_acc)

class MyStacking(StakingModel):
    def eval(self, eval_data, daily_acc = False):

        cap = self.lr_model.cap

        _, rmse_nanwang,guowang_rmse = solar_wind_mse(eval_data['actual'], eval_data['predict_load'],
                       cap)
        _mse = acc_mse(eval_data['actual'], eval_data['predict_load'],
                       cap, 0)
        _mape = MAPE(eval_data['actual'], eval_data['predict_load'],cap)
        if daily_acc:
            daily_mse_mape(eval_data, cap)
        print(f"test rmse_nanwang:{rmse_nanwang}")
        return guowang_rmse, _mape

    def test(self, test_data):
        # actual = test_data.loc[:, [self.load_name, self.time_name]]
        # predict_data = test_data.drop(columns=[self.load_name])
        predict_result = self.predict(test_data)
        predict_result["actual"] = test_data[self.load_name].values
        predict_result.rename(columns={"date":"datetime"}, inplace=True)
        _mse, _mape = self.eval(predict_result, daily_acc=True)
        logger.info(f"final test guowang rmse:{_mse}, mape:{_mape}")
        return predict_result

class ConnectStacking(MyStacking):

    def train(self, train_data):
        """
        n折训练数据
        :param train_data:
        :return:
        """
        # 挑选特征
        feature_list = self.models[0].config.get_para("feature_list")
        feature_list = list(set(feature_list + [self.models[0].load_name, self.models[0].time_name]))
        train_data = train_data.loc[:, feature_list]
        train_data.reset_index(drop=True, inplace=True)

        # 归一化
        _feature_list = [n for n in feature_list if n not in [self.models[0].load_name, self.models[0].time_name]]
        norm_feature, norm_label = self.normalize(train_data.loc[:, _feature_list], train_data[self.models[0].load_name])
        norm_train_data = pd.concat([norm_feature, norm_label], axis=1)
        # norm_train_data = pd.concat([train_data.loc[:, _feature_list], norm_label], axis=1)
        norm_train_data[self.models[0].time_name] = train_data[self.models[0].time_name].values

        test_result = pd.DataFrame()
        acc_df = pd.DataFrame()

        add_feature = []

        for i in range(len(self.models)):
            m = self.models[i]
            if len(add_feature):
                feature_list = self.models[i].config.get_para("feature_list")
                feature_list += add_feature
                self.models[i].config.input_para("feature_list", feature_list)

            _test_result, mse_df = self.models[i].k_fold_train(norm_train_data, k_fold=self.k_fold)
            _test_result.index = pd.to_datetime(_test_result["date"])

            norm_train_data[f"{m.model_name}_pred"] = _test_result[f"{m.model_name}_pred"].values
            add_feature.append(f"{m.model_name}_pred")

            if not len(test_result):
                test_result = _test_result.copy()
                acc_df = mse_df.copy()
                continue
            test_result[f"{m.model_name}_pred"] = _test_result.loc[test_result.index, f"{m.model_name}_pred"].values
            acc_df = pd.merge(left=acc_df, right=mse_df, on='k', how='left')

        final_feature = self.models[-1].config.get_para("feature_list")
        final_feature = list(set(final_feature + [self.load_name, self.time_name] + add_feature))
        _test_result, mse_df = self.lr_model.k_fold_train(norm_train_data[final_feature].rename(columns={self.load_name:"actual"}),
                                                          k_fold=self.k_fold)

        # _test_result, mse_df = self.lr_model.k_fold_train(test_result, k_fold=self.k_fold)
        _test_result['predict_load'] = _test_result[f"{self.lr_model.model_name}_pred"]

        _test_result['predict_load'] = self.denormalization(_test_result['predict_load']).reshape(-1)
        _test_result['actual'] = self.denormalization(_test_result['actual']).reshape(-1)

        dir_path = os.path.join(self.models[0].area_path, self.area_name, "model")
        feature_path = os.path.join(dir_path, "MINMAX_LABEL.csv")
        feature_data = pd.read_csv(feature_path)
        self.lr_model.cap *= feature_data.loc[0, '0']
        for m in self.models:
            m.cap *= feature_data.loc[0, '0']

        _mse, _mape = self.eval(_test_result)
        logger.info(f"final mse:{_mse}, mape:{_mape}")
        return _test_result

    def predict(self, predict_data):

        norm_predict_data = self.normalize_feature(predict_data.drop(columns=self.models[0].time_name))
        norm_predict_data[self.models[0].time_name] = predict_data[self.models[0].time_name].values
        # norm_predict_data = predict_data.copy()
        all_fold_predict = pd.DataFrame()
        all_model_name = []
        add_feature = []
        for i in range(len(self.models)):
            _model = self.models[i]
            logger.info(f"model {_model.model_name} predict")
            if len(add_feature):
                feature_list = _model.config.get_para("feature_list")
                feature_list += add_feature
                feature_list = list(set(feature_list))
                self.models[i].config.input_para("feature_list", feature_list)
            all_model_name.append(_model.model_name)

            _predict_result = self.models[i].k_fold_predict(norm_predict_data, self.k_fold)
            all_fold_predict[_model.model_name] = _predict_result[_model.model_name].values
            norm_predict_data[f"{_model.model_name}_pred"] = all_fold_predict[_model.model_name].values
            add_feature.append(f"{_model.model_name}_pred")
        final_feature = self.models[-1].config.get_para("feature_list")
        final_feature = list(set(final_feature+[self.time_name]+add_feature))
        # lr_predict_result = self.lr_model.k_fold_predict(all_fold_predict, self.k_fold)
        lr_predict_result = self.lr_model.k_fold_predict(norm_predict_data[final_feature], self.k_fold)

        predict_result = pd.DataFrame()
        predict_result["predict_load"] = self.denormalization(lr_predict_result['lr'].values).reshape(-1)
        predict_result["date"] = predict_data[self.models[0].time_name].values

        return predict_result

def get_Model(config, model_names=[]):
    """
    实例化模型
    :param config:
    :return:
    """
    models = []

    model_dic = {
        "xgboost":  MyXGB,
        "lightgbm": MyLGB,
        "lstm":     MyLSTM,
        "lasso":    MyLasso,
        "knn":      MyKnn,
        "cnn":      MyCNN,
                 }
    if not len(model_names):
        model_names = config.get_para("stack_model_list")
    for model_name in model_names:
        MODEL = model_dic[model_name]
        model = MODEL(config)
        models.append(model)
    return models



def run_lgb(area_name, is_train=True):
    config = config_parser(area_name)
    # config.input_para("feature_list", config.get_para("wind_direction_feas") + config.get_para("clean_feature_cols"))
    area_path = os.path.join("./","data", "area")
    cap = config.get_para("capacity")

    lgb_model = MyLGB(config)
    xgb_model = MyXGB( config)
    data = data_generator(config, True, split=False)
    data["date"] = pd.to_datetime(data["date"])
    predict_data = data[data["date"] >= pd.to_datetime("2022-8-1")]
    if is_train:
        train_data = data.drop(index=predict_data.index)
        xgb_model.train(train_data)

    predict_result, _mse, _mape = xgb_model.test(predict_data)

    # predict_result = stacking_model.test(predict_data)
    # save_name = ""
    # for m in stacking_model.models:
    #     save_name += f"{m.model_name}_"
    # save_name += "lr.csv"
    #
    save_path = os.path.join(area_path, area_name, "result", "xgb_test_result.csv")
    predict_result.to_csv(save_path, index=False)


def run_train(config, is_train=True):
    """
    运行训练主程序
    :param area_name:
    :param is_train:
    :return:
    """
    area_name = config.area_name
    area_path = os.path.join("./", "data", "area")
    try:
        data = data_generator(config, True, split=False)
    except:
        train_logger.info('problems with data,unable to train')
        return np.nan,np.nan  
    data["date"] = pd.to_datetime(data["date"])
    
    ####数据量校核
    if len(data)==0:
        train_logger.info('Zero data,unable to train')
        return np.nan,np.nan   
    if len(data)//96<20:
        train_logger.info('few data,unable to train')
        return np.nan,np.nan 
    ####数据时间格式校核
    from center.tools.common import check_data,fix_date
    data=fix_date(data,time_col="date",freq="15T",start="00:15",end="00:00",fillna=True)
    check_data(data,'date',day_len=96)

    #######数据质量校核
    # from check_source_data import check_quality
    # feas_list=config.get_para('feature_list')
    # good_data, bads=check_quality(data,feas_list)
    # train_logger.info(f'bad feas:{bads}')
    # if not good_data:
    #     train_logger.info(f'{area_name}数据没有通过数据质量检测!')
    #     return np.nan,np.nan   
    
    ########数据清洗
    # cleaner=Clean(station_type=config.station_info_dict['station_type'],
    #               use_type='train',
    #               station_name=config.area_name,
    #               capacity=config.station_info_dict['capacity'],
    #               freq=config.station_info_dict['day_point'],
    #               longitude=config.station_info_dict['longitude'],
    #               latitude=config.station_info_dict['latitude'],
    #               similarity_detect= False,
    #               threshold=config.station_info_dict['threshold']
    #                 )
    # feas=config.get_para("feature_list")+[config.station_info_dict['load_name']]
    # data_cleaned,del_info = cleaner.clean_station(
    #                             data,
    #                             clean_col_list=feas,
    #                             time_col=config.station_info_dict['time_name'],
    #                             load_col=config.station_info_dict['load_name'])
    
    # from dataclean import fix_date,plot_peroid
    # data_cleaned=fix_date(data_cleaned,data,freq='5T')
    # plot_peroid(data,'before',time_col = "date",cols = feas,start_day = "2022-9-24",end_day=None,days = 10)
    # plot_peroid(data_cleaned,'after',time_col = "date",cols = feas,start_day = "2022-9-24",end_day=None,days = 10)   
    
    # from center.tools.common import check_data,fix_date
    # data_cleaned=fix_date(data_cleaned,time_col="date",freq="15T",start="00:00",end="23:45",fillna=True)
    # check_data(data_cleaned,'date',day_len=96)
    
    # data_cleaned=data_cleaned.dropna()
    # predict_data=data_cleaned.iloc[-96*14:,:]
    predict_data=data.iloc[-96*7:,:]
    model_combine = config.get_para("model_combine")

    # 使用单个模型
    if not model_combine:
        single_model = config.get_para("select_algo")
        print(f"train single model :{single_model}")
        model = singleLSTM(config) if single_model.lower()=="lstm" else get_Model(config, [single_model])[0]
        if is_train:
            train_data = data.drop(index=predict_data.index)
            model.train(train_data)
        predict_result, _mse, _mape = model.test(predict_data)
        predict_result.to_csv(model.test_file, index=False)
        return _mse, _mape


    k_fold = config.get_para("fold_num")
    lr_model = Mylinear(config)
    models = get_Model(config)
    stacking_model = MyStacking(area_name, models, lr_model, k_fold=k_fold)

    if is_train:
        train_data = data.drop(index=predict_data.index)
        train_result = stacking_model.train(train_data)
        save_path = os.path.join(area_path, area_name, "result", "train_result.csv")
        train_result.to_csv(save_path, index=False)

    predict_result = stacking_model.test(predict_data)
    save_name = ""
    for m in stacking_model.models:
        save_name += f"{m.model_name}_"
    save_name += "lr.csv"

    save_path = os.path.join(area_path, area_name, "result", save_name)
    predict_result.to_csv(save_path, index=False)

    _mse, _mape = test_eval(predict_result, lr_model.cap, False)
    return _mse, _mape

def run_predict_simple(area_name, predict_data):
    """
    运行预测函数
    :param area_name:
    :param predict_date:
    :return:
    """
    config = config_parser(area_name)
    area_path = os.path.join("./", "data", "area")
    cap = config.get_para("capacity")
    k_fold = config.get_para("fold_num")

    # 1.初始化模型
    # lr_model = Mylinear(area_name, config, area_path,
    #                     load_name="actual", time_name="date",
    #                     cap=cap)
    lr_model = Mylinear(config)
    models = get_Model(config)
    stacking_model = MyStacking(area_name, models, lr_model, k_fold=k_fold)

    # 2.预测
    predict_result = stacking_model.predict(predict_data)
    predict_result.rename(columns={"predict_load": "load"}, inplace=True)

    station_type = config.get_para('station_type')
    if station_type == "pv":
        predict_result["date"] = pd.to_datetime(predict_result["date"])
        # 根据配置文件，实例化数据清洗对象
        cleaner = get_cleaner(config, use_type="test")
        # 日落时间段强制置零
        predict_result.reset_index(drop=True, inplace=True)
        predict_result = cleaner.sunset_zero(data=predict_result, cols=["load"])
        predict_result.reset_index(drop=True, inplace=True)
        predict_result = solve_pv_frog_leg(predict_result, points=12)
    predict_result["load"] = predict_result["load"].apply(lambda x:x if x>0 else np.nan)
    predict_result["load"].interpolate(inplace=True)
    predict_result.fillna(method="bfill", inplace=True)
    predict_result.fillna(method="ffill", inplace=True)
    return predict_result

def run_predict(area_name):
    """
    运行预测函数
    :param area_name:
    :param predict_date:
    :return:
    """
    config = config_parser(area_name)
    k_fold = config.get_para("fold_num")

    # 1.生成预测集
    predict_data = data_generator(config, False, "", split=False)

    # 2.初始化模型
    model_combine = config.get_para("model_combine")
    if model_combine:
        print(f"stacking model predict")
        lr_model = Mylinear(config)
        models = get_Model(config)
        stacking_model = MyStacking(area_name, models, lr_model, k_fold=k_fold)

        # 3.预测
        predict_result = stacking_model.predict(predict_data)
    else:
        single_model = config.get_para("select_algo")
        print(f"single model:{single_model} predict")
        model = singleLSTM(config) if single_model.lower()=="lstm" else get_Model(config, [single_model])[0]
        # 3.预测
        predict_result = model.predict(predict_data)
    #predict_result.rename(columns={"predict_load": "load"}, inplace=True)

    station_type = config.get_para('station_type')
    if station_type == "pv":
        predict_result["date"] = pd.to_datetime(predict_result["date"])
        # 根据配置文件，实例化数据清洗对象
        cleaner = get_cleaner(config, use_type="test")
        # 日落时间段强制置零
        predict_result.reset_index(drop=True, inplace=True)
        predict_result = cleaner.sunset_zero(data=predict_result, cols=["load"])
        predict_result.reset_index(drop=True, inplace=True)
        # predict_result = solve_pv_frog_leg(predict_result, points=12)
        predict_result = predict_result.loc[:, ["date", "predict_load"]]
    predict_result["predict_load"] = predict_result["predict_load"].apply(lambda x: x if x >= 0 else np.nan)
    predict_result["predict_load"].interpolate(inplace=True)
    predict_result.fillna(method="bfill", inplace=True)
    predict_result.fillna(method="ffill", inplace=True)
    return predict_result

if __name__ == "__main__":
    import argparse
    from config import global_config, area_list
    from check_source_data import check_station
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station_name",
        type=str,
        default="all",
        help="name of one station('NARI-19008-Xibei-dtqlyf') to predict",
    )

    parser.add_argument(
        "--train",
        default=False,
        action='store_true',
        help="download data or not",
    )
    args = parser.parse_args()
    is_train = args.train
    train_logger = get_logger("train")    
    
    ####确定待处理站
    if args.station_name == "all":
        stations_list = os.listdir(os.path.join(global_config.work_path, "data", "area"))
        station_names = [n for n in area_list if n in stations_list]

    ####初始化日志
    # if os.path.exists(os.path.join(project_path,'logs/train_debug.log')):
    #     os.remove(os.path.join(project_path,'logs/train_debug.log'))
        
    ####遍历厂站，开始训练
    mape_list=[] ##保存结果用的
    #station_names=['01113304000004']
    for station_name in station_names:
        
        train_logger.info(f'################## {station_name}')
        config = config_parser(station_name)
        
        ####确保目录结构正确
        _dir=os.path.join(config.area_path,config.area_name)
        for d in ['data','model','result']:
            if not os.path.exists(os.path.join(_dir,d)):
                os.mkdir(os.path.join(_dir,d))
        
        if check_station(config): ##该站通过了检查
            st=time.time()
            _mse, _mape = run_train(config, is_train)
            print('training time:',time.time()-st)
            train_logger.info(f'{station_name} test mse:{_mse} mape:{_mape}\n')
            mape_list.append(_mape)
        else:
            train_logger.info(f'{station_name}文件有问题，没通过check_station检测\n')
            mape_list.append(np.nan)
           
