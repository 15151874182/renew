from renew_center.tools.logger import setup_logger
from renew_center.tools.plot_view import plot_peroid
from renew_center.dataclean.renew_clean import Clean
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

logger = setup_logger('logger')
# =============================================================================


class Renew():
    def __init__(self, project_path):
        self.project_path = project_path
        self.feas_used = []  # final feas used to train and predict model
    # define my zscore function which can use feas_used

    def zscore(self, data, mode, feas_used):
        df = deepcopy(data)  # not to affect origin data
        if mode == 'fit_transform':
            self.u = df.mean()
            self.std = df.std()
            df = (df-self.u)/self.std
        elif mode == 'transform':
            df = (df-self.u)/self.std
        elif mode == 'feas_used_transform':
            u_selected = self.u[feas_used]
            std_selected = self.std[feas_used]
            scaled_data = (df[feas_used]-u_selected)/std_selected
        return scaled_data

    # customerize for different project
    def load_data(self, config):
        self.real_load = pd.read_csv(os.path.join(
            config.area_path, 'data', 'history_cleaned_real_load.csv'), index_col='date', parse_dates=True)
        self.fore_wea = pd.read_csv(os.path.join(
            config.area_path, 'data', 'history_cleaned_XXL_nwp_weather.csv'), index_col='date', parse_dates=True)
        self.daily_wea = pd.read_csv(os.path.join(
            config.area_path, 'data', 'daily_cleaned_pred_weather.csv'), index_col='date', parse_dates=True)
        self.data = self.fore_wea.join(self.real_load).dropna()
        self.data.rename(columns={'GlobalR': 'rad_GlobalR', 'AirT': 'temp',
                                  'DirectR': 'rad_DirectR', 'RH': 'hum'}, inplace=True)

    def data_clean(self, data, config):
        df = deepcopy(data)  # not to affect origin data
        cleaner = Clean()
        cleaned_data = cleaner.clean(data, config)
        cleaned_data = cleaner.clean_area(data, config)
        return cleaned_data

    def make_label(self, data):
        pass

    def data_split_scale(self, data):
        data2 = deepcopy(data)  # not to affect origin data
        y = data2['y']
        del data2['y']
        x = data2
        # split dataset into train,val,test and scale
        x_trainval, self.x_test, y_trainval, self.y_test = train_test_split(
            x, y, random_state=123, shuffle=True, test_size=0.2)  # split test first
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_trainval, y_trainval, random_state=123, shuffle=True, test_size=0.2)  # then split val
        self.x_train_scaled = self.zscore(
            self.x_train, verbose='fit_transform', feas_used=None)
        self.x_val_scaled = self.zscore(
            self.x_val, verbose='transform', feas_used=None)
        self.x_test_scaled = self.zscore(
            self.x_test, verbose='transform', feas_used=None)

    def feature_engineering(self, data):
        # lagged returns feature
        for i in range(1, 3):
            data['{}-{}'.format('return', i)] = data['return'].shift(i)
        # change feature 'return' into 0,1
        data['return'] = data['return'].apply(lambda x: 1 if x >= 0.25 else 0)
        # time discretization feature
        data['date'] = data.index
        data['DayOfWeek'] = data['date'].apply(
            lambda x: x.dayofweek / 6.0 - 0.5)
        data['DayOfMonth'] = data['date'].apply(
            lambda x: (x.day - 1) / 30.0 - 0.5)
        data['DayOfYear'] = data['date'].apply(
            lambda x: (x.dayofyear - 1) / 365.0 - 0.5)
        del data['date']
        # volality feature
        data['O-C'] = data['open'] - data['close']
        data['H-L'] = data['high'] - data['low']
        # price change over 7,30 days
        for i in [7, 30]:
            data['{}{}'.format('momentum', i)] = data['close'] - \
                data['close'].shift(i)
        # moving average over 5,10,30,60 days
        for i in [5, 10, 30, 60]:
            data['{}{}'.format('MA', i)] = df['close'].rolling(window=i).mean()
        data = data.dropna()
        # Exponential MA
        data['EMA'] = data['close'].ewm(
            alpha=2/(len(data) + 1), adjust=False).mean()
        data = data.dropna()
        return data

    def subset_selection(self, model):
        # filter method
        from sklearn.feature_selection import SelectKBest, f_classif
        # Use correlation coefficients to select most relevant features
        selector = SelectKBest(score_func=f_classif,
                               k=int(len(self.x_train.columns)*0.8))
        x_selected = selector.fit_transform(self.x_train_scaled, self.y_train)
        feature_indices = selector.get_support(indices=True)
        filter_feas_used = self.x_train.columns[feature_indices]
        print('filter_feas_used:', filter_feas_used)

        # wrapper method use  Backward Elimination
        model.fit(self.x_train_scaled, self.y_train)
        y_val_pred = model.predict_proba(self.x_val_scaled)[:, 1]
        best_score = roc_auc_score(self.y_val, y_val_pred)
        wrapper_feas_used = list(self.x_train.columns)  # init fea_list
        for fea in tqdm(self.x_train.columns):
            wrapper_feas_used.remove(fea)
            model.fit(self.zscore(self.x_train[wrapper_feas_used],
                                  verbose='feas_used_transform', feas_used=wrapper_feas_used), self.y_train)
            y_val_pred = model.predict_proba(self.zscore(
                self.x_val[wrapper_feas_used], verbose='feas_used_transform', feas_used=wrapper_feas_used))[:, 1]
            score = roc_auc_score(self.y_val, y_val_pred)
            if score < best_score:  # if remove this fea and score decrease means this fea should remain
                wrapper_feas_used.append(fea)  # recovery this fea
            else:
                best_score = score  # update best_score
        print('wrapper_feas_used:', wrapper_feas_used)

        # Embedded method use Lasso
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.001)
        lasso.fit(self.x_train_scaled.values, self.y_train)
        embedded_feas_used = self.x_train.columns[lasso.coef_ != 0]
        print('embedded_feas_used:', embedded_feas_used)

        return list(filter_feas_used), list(wrapper_feas_used), list(embedded_feas_used)

    def finetune(self, model, feas_used, method='optuna', n_trials=100):

        # use optuna lib to finetune SVC hyperparameters
        if method == 'optuna':
            import optuna

            def objective(trial):
                # Define hyperparameter Search Scope
                C = trial.suggest_loguniform(
                    'C', 0.01, 1.0)  # range from 0.1 to 10
                kernel = trial.suggest_categorical('kernel', ['poly', 'rbf'])
                gamma = trial.suggest_loguniform(
                    'gamma', 0.01, 1.0)  # range from 0.1 to 1.0
                model = SVC(probability=True, C=C, kernel=kernel, gamma=gamma)
                model.fit(self.x_train_scaled[feas_used], self.y_train)
                y_val_pred = model.predict_proba(
                    self.x_val_scaled[feas_used])[:, 1]
                auc = roc_auc_score(self.y_val, y_val_pred)
                return auc
            study = optuna.create_study(
                direction='maximize')  # maximize the auc
            study.optimize(objective, n_trials=n_trials)
            print("Best parameters:", study.best_params)
            best_model = SVC(probability=True, **study.best_params)
            best_model.fit(self.x_train_scaled[feas_used], self.y_train)
            y_test_pred = best_model.predict_proba(
                self.x_test_scaled[feas_used])[:, 1]
            test_score = roc_auc_score(self.y_test, y_test_pred)
            print('test_score:', test_score)
        elif method == 'gridsearch':
            # use gridsearch finetune hyperparameters
            param_grid = {
                'C': list(np.linspace(0.1, 2.0, 5)),
                'kernel': ['rbf', 'poly'],
                'gamma': list(np.linspace(0.01, 1.0, 5))
            }
            grid_search = GridSearchCV(estimator=SVC(
                probability=True), param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2)
            grid_search.fit(self.x_train_scaled[feas_used], self.y_train)
            print("Best parameters:", grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            best_model = grid_search.best_estimator_
            y_test_pred = best_model.predict_proba(
                self.x_test_scaled[feas_used])[:, 1]
            test_score = roc_auc_score(self.y_test, y_test_pred)
            print('test_score:', test_score)
        return best_model

    def investigate_model(self, model, feas_used, name='tuned_SVC'):
        # Display confussion matrix
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            self.x_test_scaled[feas_used],
            self.y_test,
            display_labels=model.classes_,
            cmap=plt.cm.Blues
        )
        disp.ax_.set_title('Confusion matrix')
        plt.show()

        # Classification Report
        y_test_pred = model.predict(self.x_test_scaled[feas_used])
        print(classification_report(self.y_test, y_test_pred))

        # Display ROCCurve
        disp_roc = RocCurveDisplay.from_estimator(
            model,
            self.x_test_scaled[feas_used],
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
        default="all",  # all for all areas // NARI-19008-Xibei-dtqlyf,NARI-19008-Xibei-dtqlyf for 2 areas
        help="name of areas to predict, 1, more, all areas both ok",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",  # train/predcit
        help="train/predcit",
    )
    args = parser.parse_args()
# =============================================================================
    # get areas
    from config import EnvConfig, AreaConfig
    project_path = EnvConfig().project_path
    if args.area == "all":
        areas = os.listdir(os.path.join(project_path, "data", "area"))
    else:
        areas = args.area.split(',')  # list type
# =============================================================================
    renew = Renew(project_path)
    config = AreaConfig('NARI-19001-Xibei-abhnfd')
    renew.load_data(config)
    if len(renew.data)//96 < 30:
        logger.warning('len(data)//96<30')

    if args.mode == 'train':
        setattr(config, 'mode', 'train')
        setattr(config, 'data', renew.data)
        cleaner = Clean()
        # cleaned_data = cleaner.clean(data=config.data,
        #                              feas_used=list(renew.data.columns),
        #                              area='xxx',
        #                              area_type=config.area_type,
        #                              # 'train'(may delete bad feas)/'predcit'(fix data only)
        #                              capacity=config.capacity,
        #                              online=True,  # None for wind clean
        #                              Longitude=config.Longitude,  # None for wind clean
        #                              Latitude=config.Latitude,  # None for wind clean
        #                              # 3 inputs only, compare 2 feas' trend to clean, 0.8 means remain top 80% data, False if unwanted
        #                              trend=['load', 'rad', 0.8],
        #                              plot=[['load', 'rad'], "2022-03-19", 30])
        
        cleaned_data = cleaner.clean_area(config,online=True,plot=[['load', 'speed_30'], "2022-06-19", 30])
    elif args.mode == 'predict':
        setattr(config, 'mode', 'test')

