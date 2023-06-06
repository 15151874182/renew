import os
import sys
import time
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import traceback
import logging
import warnings
warnings.filterwarnings('ignore') 

from renew_clean import Clean
# =============================================================================
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
logger = logging.getLogger('logger')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG) 
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler("./logs/logger.log")
file_handler.setLevel(logging.DEBUG) 
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
'''
logging.info("info")
logging.debug("debug")
logging.warning("warning")
logging.error("error")
logging.critical("critical")
'''

class Renew(): 
    def __init__(self,project_path):
        self.project_path=project_path
    
    ##define my zscore function which can use selected_feas
    def zscore(self,data,mode,selected_feas): 
        df=deepcopy(data) ##not to affect origin data
        if mode=='fit_transform':
            self.u=df.mean()
            self.std=df.std()
            df=(df-self.u)/self.std   
        elif mode=='transform':
            df=(df-self.u)/self.std 
        elif mode=='selected_feas_transform':
            u_selected=self.u[selected_feas]
            std_selected=self.std[selected_feas]
            scaled_data=(df[selected_feas]-u_selected)/std_selected 
        return scaled_data
    
    ##customerize for different project
    def load_data(self,config):
        self.real_load=pd.read_csv(os.path.join(config.area_path,'data','history_cleaned_real_load.csv'),index_col=0,parse_dates=True)
        self.fore_wea=pd.read_csv(os.path.join(config.area_path,'data','history_cleaned_XXL_nwp_weather.csv'),index_col=10,parse_dates=True)
        self.daily_wea=pd.read_csv(os.path.join(config.area_path,'data','daily_cleaned_pred_weather.csv'),index_col=10,parse_dates=True)
        self.data=self.fore_wea.join(self.real_load).dropna() 
    
    def data_clean(self,data,config):
        df=deepcopy(data) ##not to affect origin data
        cleaner=Clean()
        cleaned_data=cleaner.clean(data,config)
        cleaned_data=cleaner.clean_area(data,config)
        return cleaned_data
    
    def make_label(self,data):
        pass

    def data_split_scale(self,data):
        data2=deepcopy(data) ##not to affect origin data
        y=data2['y']
        del data2['y']
        x=data2
        ####split dataset into train,val,test and scale
        x_trainval,self.x_test,y_trainval,self.y_test=train_test_split(x,y,random_state=123,shuffle=True,test_size=0.2) ##split test first
        self.x_train,self.x_val,self.y_train,self.y_val=train_test_split(x_trainval,y_trainval,random_state=123,shuffle=True,test_size=0.2) ##then split val 
        self.x_train_scaled = self.zscore(self.x_train,verbose='fit_transform',selected_feas=None)
        self.x_val_scaled = self.zscore(self.x_val,verbose='transform',selected_feas=None)
        self.x_test_scaled = self.zscore(self.x_test,verbose='transform',selected_feas=None)
    
    def feature_engineering(self,data):
        ####lagged returns feature
        for i in range(1,3):
            data['{}-{}'.format('return',i)]=data['return'].shift(i)    
        ####change feature 'return' into 0,1
        data['return']=data['return'].apply(lambda x: 1 if x>=0.25 else 0)
        ####time discretization feature
        data['date']=data.index
        data['DayOfWeek']=data['date'].apply(lambda x : x.dayofweek / 6.0 - 0.5)
        data['DayOfMonth']=data['date'].apply(lambda x : (x.day - 1) / 30.0 - 0.5)
        data['DayOfYear']=data['date'].apply(lambda x : (x.dayofyear - 1) / 365.0 - 0.5)
        del data['date']
        ####volality feature
        data['O-C'] = data['open'] - data['close']
        data['H-L'] = data['high'] - data['low']
        ####price change over 7,30 days
        for i in [7,30]:
            data['{}{}'.format('momentum',i)]=data['close']-data['close'].shift(i)
        ####moving average over 5,10,30,60 days
        for i in [5,10,30,60]:
            data['{}{}'.format('MA',i)]=df['close'].rolling(window=i).mean()
        data=data.dropna()   
        ####Exponential MA
        data['EMA'] = data['close'].ewm(alpha=2/(len(data) + 1), adjust=False).mean()
        data=data.dropna()   
        return data
    
    def subset_selection(self,model):
        ####filter method
        from sklearn.feature_selection import SelectKBest,f_classif
        #### Use correlation coefficients to select most relevant features
        selector = SelectKBest(score_func=f_classif, k=int(len(self.x_train.columns)*0.8))        
        x_selected = selector.fit_transform(self.x_train_scaled, self.y_train)
        feature_indices = selector.get_support(indices=True)
        filter_selected_feas = self.x_train.columns[feature_indices]
        print('filter_selected_feas:',filter_selected_feas)        
        
        ####wrapper method use  Backward Elimination
        model.fit(self.x_train_scaled, self.y_train)
        y_val_pred=model.predict_proba(self.x_val_scaled)[:, 1]
        best_score = roc_auc_score(self.y_val,y_val_pred)   
        wrapper_selected_feas=list(self.x_train.columns) ##init fea_list
        for fea in tqdm(self.x_train.columns):
            wrapper_selected_feas.remove(fea) 
            model.fit(self.zscore(self.x_train[wrapper_selected_feas],verbose='selected_feas_transform',selected_feas=wrapper_selected_feas),self.y_train)
            y_val_pred=model.predict_proba(self.zscore(self.x_val[wrapper_selected_feas],verbose='selected_feas_transform',selected_feas=wrapper_selected_feas))[:, 1]    
            score = roc_auc_score(self.y_val,y_val_pred)
            if score<best_score: ##if remove this fea and score decrease means this fea should remain
                wrapper_selected_feas.append(fea) ##recovery this fea    
            else:
                best_score=score ##update best_score
        print('wrapper_selected_feas:',wrapper_selected_feas)
        
        ####Embedded method use Lasso
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.001)  
        lasso.fit(self.x_train_scaled.values, self.y_train)
        embedded_selected_feas = self.x_train.columns[lasso.coef_ != 0]
        print('embedded_selected_feas:',embedded_selected_feas)
        
        return list(filter_selected_feas), list(wrapper_selected_feas), list(embedded_selected_feas)
    
    def finetune(self,model,selected_feas,method='optuna',n_trials=100):
        
        #### use optuna lib to finetune SVC hyperparameters
        if method=='optuna':
            import optuna
            def objective(trial):
                # Define hyperparameter Search Scope
                C = trial.suggest_loguniform('C', 0.01, 1.0)  ##range from 0.1 to 10
                kernel = trial.suggest_categorical('kernel', ['poly', 'rbf'])
                gamma = trial.suggest_loguniform('gamma', 0.01, 1.0) ##range from 0.1 to 1.0
                model = SVC(probability=True,C=C, kernel=kernel, gamma=gamma)
                model.fit(self.x_train_scaled[selected_feas], self.y_train)
                y_val_pred = model.predict_proba(self.x_val_scaled[selected_feas])[:, 1]
                auc = roc_auc_score(self.y_val, y_val_pred)
                return auc
            study = optuna.create_study(direction='maximize') ##maximize the auc
            study.optimize(objective, n_trials=n_trials)
            print("Best parameters:", study.best_params)
            best_model = SVC(probability=True,**study.best_params)
            best_model.fit(self.x_train_scaled[selected_feas], self.y_train)
            y_test_pred = best_model.predict_proba(self.x_test_scaled[selected_feas])[:, 1]
            test_score = roc_auc_score(self.y_test,y_test_pred)
            print('test_score:',test_score)        
        elif method=='gridsearch':
            #### use gridsearch finetune hyperparameters
            param_grid = {
                'C': list(np.linspace(0.1,2.0,5)),
                'kernel': ['rbf','poly'],
                'gamma': list(np.linspace(0.01,1.0,5))
            }
            grid_search = GridSearchCV(estimator=SVC(probability=True), param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2)
            grid_search.fit(self.x_train_scaled[selected_feas], self.y_train)
            print("Best parameters:", grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            best_model = grid_search.best_estimator_
            y_test_pred = best_model.predict_proba(self.x_test_scaled[selected_feas])[:, 1]
            test_score = roc_auc_score(self.y_test,y_test_pred)
            print('test_score:',test_score)
        return best_model
    
    def investigate_model(self,model,selected_feas,name='tuned_SVC'):
        # Display confussion matrix
        disp = ConfusionMatrixDisplay.from_estimator(
                model,
                self.x_test_scaled[selected_feas],
                self.y_test,
                display_labels=model.classes_,
                cmap=plt.cm.Blues
            )
        disp.ax_.set_title('Confusion matrix')
        plt.show()
        
        # Classification Report
        y_test_pred = model.predict(self.x_test_scaled[selected_feas])
        print(classification_report(self.y_test, y_test_pred))

        # Display ROCCurve 
        disp_roc = RocCurveDisplay.from_estimator(
                    model, 
                    self.x_test_scaled[selected_feas],
                    self.y_test,
                    name=name)
        disp_roc.ax_.set_title('ROC Curve')
        plt.show()

# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area",
        type=str,
        default="all", ##all for all areas // NARI-19008-Xibei-dtqlyf,NARI-19008-Xibei-dtqlyf for 2 areas
        help="name of areas to predict, 1, more, all areas both ok",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train", ##train/predcit
        help="train/predcit",
    )
    args = parser.parse_args()
# =============================================================================    
    ####get areas
    from config import EnvConfig, AreaConfig
    project_path=EnvConfig().project_path
    if args.area == "all":
        areas = os.listdir(os.path.join(project_path, "data", "area"))
    else:
        areas=args.area.split(',') ##list type
# =============================================================================            
    renew=Renew(project_path)
    config=AreaConfig('NARI-19012-Xibei-gbdyfd')
    renew.load_data(config)
    if args.mode=='train':
        setattr(config, 'mode', 'train')
    elif args.mode=='predict':
        setattr(config, 'mode', 'test')


    # mape_list=[] 
    # for station_name in station_names:
        
    #     train_logger.info(f'################## {station_name}')
    #     config = config_parser(station_name)
        
    #     ####确保目录结构正确
    #     _dir=os.path.join(config.area_path,config.area_name)
    #     for d in ['data','model','result']:
    #         if not os.path.exists(os.path.join(_dir,d)):
    #             os.mkdir(os.path.join(_dir,d))
        
    #     if check_station(config): ##该站通过了检查
    #         st=time.time()
    #         _mse, _mape = run_train(config, is_train)
    #         print('training time:',time.time()-st)
    #         train_logger.info(f'{station_name} test mse:{_mse} mape:{_mape}\n')
    #         mape_list.append(_mape)
    #     else:
    #         train_logger.info(f'{station_name}文件有问题，没通过check_station检测\n')
    #         mape_list.append(np.nan)
           
