import pandas as pd
import numpy as np
from dateutil import relativedelta
pd.options.display.max_rows = 100
from datetime import datetime
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')




class Feature:
    # DF = pd.read_csv('../data/data_vf Copy.csv')
    # DF['usd_pnl_ebit'] = DF['usd_pnl_ebitda'] + DF['usd_pnl_depreciationAndAmortization']
    COLS_BASE = ['symbol' , 'companyName' , 'industry' ,  'industry-category' , 'sector', 'country','month' , 'year' , 'datetime', 'date' , 'cal_period', 'cal_yearperiod']
    COLS_PRICE = ['usd_adjClose' , 'usd_marketCap', 'marketCap_log']

    def __init__(self, df):
        self.df = df
        print('Feature object instantiated')

    def df_load(self, df=None):
        if df is None:
            colls_all = self.cols('all')
            df2 = self.df[colls_all]
            print(f'Default DataFrame loaded with size {df2.shape[0]:.0f} rows : {df2.shape[1]:.0f} columns')
            return df2
        elif not isinstance(df, pd.DataFrame):
            print(f'loaded objec type{type(df)}, expected DataFrame')
            raise TypeError
        elif isinstance(df,pd.DataFrame):
            print(f'loaded DataFrame with size {df.shape[0]} rows : {df.shape[1]} columns')
            return df

    def cols(self, choice:str):
        """
        choice: "bs" , "cf" , "pnl" , "fin", "ratios", "all"
        """
        cols_bs = [i for i in self.df.columns.tolist() if i[:6]== "usd_bs"]
        cols_cf = [i for i in self.df.columns.tolist() if i[:6]== "usd_cf"]
        cols_pnl = [i for i in self.df.columns.tolist() if i[:7]=="usd_pnl"]
        cols_ratios = [i for i in self.df.columns.tolist() if i[:2] == "cf" or i[:2]== "bs" or i[:3]=="pnl"]
        if choice == "bs":
            return cols_bs
        elif choice == "pnl":
            return cols_pnl
        elif choice == "cf":
            return cols_cf
        elif choice == "ratios":
            return cols_ratios
        elif choice == "fin":
            return cols_bs+cols_pnl+cols_cf
        elif choice == "all":
            return Feature.COLS_BASE+Feature.COLS_PRICE+cols_bs+cols_pnl+cols_cf+cols_ratios
        else:
            print(f'invalid choice {choice} please select from "bs" , "cf" , "pnl" , "fin", "ratios", "all"')
            raise ValueError

    @property
    def columns(self):
        columns = self.df.columns.tolist()
        return columns

    @property
    def info(self):
        info = self.df.info(verbose=True, show_counts=True)

    def shift_manual(self, df, col_shift: str, col_calc:str ,group=None, shift=1):
        if df is None:
            print('Default DataFrame loaded')
            df = self.df[Feature.COLS_BASE]
            if group is not None:
                l_group = list(df.groupby(group))
                l2 = []
                for i in l_group:
                    idx1 = list(i[1][col_shift].sort_values(ascending=True))[shift:]
                    idx2 = list(i[1][col_shift].sort_values(ascending=True))[:-shift]
                    df_main = i[1][[col_shift , col_calc]].set_index(col_shift)
                    df_shift = df_main.loc[idx2]
                    df_shift[col_shift] = idx1
                    df_shift.reset_index(drop=True, inplace=True)
                    df_shift.set_index(col_shift, inplace=True)
                    df_main2 = df_main.merge(df_shift, how = 'left', left_index=True, right_index=True, suffixes=('_t', '_t-'+str(shift)))
                    df_main2[group] = i[0]
                    df_main2.sort_index(ascending=True)
                    l2.append(df_main2)
                    df_final = pd.concat(l2)
                    df_final.reset_index(inplace=True)
                return df_final
    
    def shift(self, df, group, col_period, periods, col_shift: list):
        for col in col_shift:
            df['key_shift'] = df[group]+"."+df[col_period]
            df.set_index('key_shift', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            # df[col+"_t-"+str(periods)] = df.groupby(group)[col].shift(periods=periods)
            col_name = col+"_t-"+str(periods)
            try:
                df.insert(loc=df.shape[1],column= col_name, value= df.groupby(group)[col].shift(periods=periods))
            except Exception as e:
                print(f'Error inserting {col_name}')
                print(f'{e}\n')
        df.reset_index(drop=False, inplace=True)
        return df

    def change(self, df:pd.DataFrame, group:str, col_period:str, periods:int, col_shift: list, log=False, drop=False):
        df = self.shift(df, group, col_period, periods, col_shift)
        for col in col_shift:
            col_name = col+"_t-"+str(periods)
            col_delta = col_name+"_delta"
            col_new = col_name+"_change"
            col_log = col_name+"_logchange"
            try:
                if log==False:
                    df[col_delta] = df[col_name] - df[col]

                    nom_zero = df[col_delta]==0
                    nom_notzero = df[df[col_delta]!=0]
                    denom_zero = df[df[col_name]==0]

                    # df.loc[denom_zero & nom_notzero, col_new] = df[col_delta] / 0.001
                    
                    # df.loc[nom_zero, col_new] = 0
                    df[col_new] = df[col_delta] / - df[col_name]
                    # df.insert(loc=df.shape[1],column= col_new, value= (df[col_name] - df[col]) / - df[col_name])
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    # df[col_new] = (df[col_name] - df[col]) / - df[col_name]
                else:
                    df[col_log]=np.log(df[col] / df[col_name])
                    # df.insert(loc=df.shape[1],column= col_log, value= np.log(df[col] / df[col_name]))
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                if drop ==True:
                    df.drop(columns = col, inplace=True)
                    df.drop(columns=col_name, inplace=True)
            except Exception as e:
                print(f'Error inserting {col_new}')
                print(f'{e}\n')
        return df

    def add_avg(self, df:pd.DataFrame = None, cols:list=None):
        """
        creates new series of comparatives with industry average and global data set average of the desired columns [cols]
        """
        if df is None:
            print('Default DataFrame loaded')
            df = self.df[Feature.COLS_BASE]
        else:
            if isinstance(cols,list):
                df = df
                print('New DataFrame assigned')
                for i in cols:
                    df['key_industry']=df['cal_yearperiod']+"."+df['industry-category']
                    df_filter = df[i].notna()
                    df_industry = df.loc[df_filter].groupby('key_industry')[i].mean().reset_index(name=i+"_indavg") ## New: remove na when calc averages
                    df_global = df.loc[df_filter].groupby('cal_yearperiod')[i].mean().reset_index(name=i+"_globalavg") ## New: remove na when calc averages
                    df = df.merge(df_industry, on = 'key_industry' , how = 'inner')
                    df = df.merge(df_global, on = 'cal_yearperiod' , how = 'inner')
                    df[i+"_excess_indavg"] = df[i] - df[i+"_indavg"]
                    df[i+"_excess_globalavg"] = df[i] - df[i+"_globalavg"]
            else:
                print(f'second argument "cols" is of type{type(cols)} expected list')
                raise TypeError
        return df

    def replacecol(df: pd.DataFrame, col_replace: str, by: list, replace=False):
        cols = df.columns.tolist()
        col_new = col_replace+"_new"
        if col_new in cols:
            df.drop(columns=col_new,inplace=True)

        index_col = df.columns.get_loc('usd_bs_cashAndShortTermInvestments')
        
        l = []
        l.append(col_replace)
        l.append(col_new)
        all_cols = l+by
        
        if replace==False:
            df.insert(index_col, col_new, df.apply(lambda row: sum([row[i] for i in by]), axis=1))
        
        else:
            df[col_replace] = df.apply(lambda row: sum([row[i] for i in by]), axis=1)
        # tot = df.apply(lambda row: sum(row(args)), axis=1)
        return df

    def ratio(self, nom, denom):
        if (denom == np.nan) or (nom == np.nan):
            return np.nan
        elif (denom == 0) & (nom != 0):
            return nom / 0.001
        elif (denom !=0 or denom != np.nan) & (denom !=0 or denom != np.nan):
            return nom / denom
            

    def add_ratio(self,df_input=None, **kwargs):
        df = df_input.copy()
        if df is None:
            print('Default DataFrame loaded')
            df = self.df[Feature.COLS_BASE]
        else:
            print('New DataFrame assigned')
            df = df
        df['key_industry']=df['cal_yearperiod']+"."+df['industry-category']
        rows = 0
        rows_tot = df.shape[0]
        for i in kwargs:
            col_ratio = i+"_ratio"
            col_indavg = i+"_ratio_indavg"
            col_globalavg = i+"_ratio_globalavg"
            print(i)
            # try:
            print(f'{col_ratio}: {kwargs[i][0]} , {kwargs[i][1]}')

            ## add new column. if denom is zero then divide by a very small number
            df[col_ratio] = df.apply(lambda row: self.ratio(row[kwargs[i][0]] , row[kwargs[i][1]]), axis = 1)
            # df.insert(loc=df.shape[1],column= i+"_ratio", value=(self.df[kwargs[i][0]] / self.df[kwargs[i][1]])) ## old ratio code
            # df[col_ratio] = np.nan
            # df.loc[self.df[kwargs[i][1]] == 0, col_ratio] = self.df[kwargs[i][0]] / 0.001
            # df.loc[self.df[kwargs[i][1]] != 0, col_ratio] = self.df[kwargs[i][0]] / self.df[kwargs[i][1]]
            # df.insert(loc=df.shape[1],column= i+"_ratio", value=(self.df[kwargs[i][0]] / self.df[kwargs[i][1]])) # old colmn insert
            rowsbefore = df.shape[0]
            
            df = df.loc[~((df[col_ratio].isna()) | (df[col_ratio] ==np.nan) | (df[col_ratio] == np.inf) | (df[col_ratio] == -np.inf) )] ## removing only no data
            # df = df.loc[~( (df[col_ratio] ==0) | (df[col_ratio].isna()) | (df[col_ratio] ==np.nan) | (df[col_ratio] == np.inf) | (df[col_ratio] == -np.inf) )] ### removing no data and zeros
            # df[col_ratio].replace([-np.inf, np.inf],0)

            rowsafter = df.shape[0]
            removed=rowsbefore-rowsafter
            rows += removed

            print(f'removed {removed:.0f} rows after {col_ratio}')
            print(f'total removed: {rows:.0f} rows, {(removed/rows_tot*100):.0f} %\n')

            # global & industry average

            ## step1: 
            df_filter = df[col_ratio].notna() ## New: remove na when calc averages
            df_industry = df.loc[df_filter].groupby('key_industry')[col_ratio].mean().reset_index(name=col_indavg) ## New: remove na when calc averages
            df_global = df.loc[df_filter].groupby('cal_yearperiod')[col_ratio].mean().reset_index(name=col_globalavg) ## New: remove na when calc averages
            
            df = df.merge(df_industry, on = 'key_industry' , how = 'inner')
            df = df.merge(df_global, on = 'cal_yearperiod' , how = 'inner')
            
            ## step2:
            df[col_ratio+"_excess_globalavg"] = df[col_ratio] - df[col_globalavg]
            df[col_ratio+"_excess_indavg"] = df[col_ratio] - df[col_indavg]
            # except Exception as e:
            #     print(e)
            #     print(f'error inserting {i}_ratio\n')
        return df

    def train(self, df=None, col_period='cal_yearperiod' , train_window=4 , test_window=1, test_gap = 0, expanding=False):
        df = self.df_load(df)
        test_train = []
        # test_train = {}

        periods = df[col_period].unique().tolist()
        periods.sort()

        if expanding == False:

            for i,j in enumerate(periods):
                if i < len(periods)-train_window-1:
                    train_beg = j
                    train_beg_idx = periods.index(train_beg)

                    train_end_idx = train_beg_idx+ train_window
                    train_end = periods[train_end_idx]
                    
                    test_beg_idx = train_end_idx+test_gap
                    test_end_idx = test_beg_idx+test_window

                    # train = periods[train_idx]
                    train_periods = periods[train_beg_idx: train_end_idx]
                    test_periods = periods[test_beg_idx: test_end_idx]

                    df_train = df.loc[df.cal_yearperiod.isin(train_periods)]
                    df_test = df.loc[df.cal_yearperiod.isin(test_periods)]

                    test_train.append((df_train, df_test))
                    # test_train[i] = {"train": df_train , "test": df_test}

        else:
            for i,j in enumerate(periods):
                if i < len(periods)-train_window-1:
                    train_beg = periods[0]
                    train_beg_idx = periods.index(train_beg)

                    train_end_idx = train_beg_idx+ train_window+i
                    train_end = periods[train_end_idx]
                    
                    test_beg_idx = train_end_idx+test_gap
                    test_end_idx = test_beg_idx+test_window

                    # train = periods[train_idx]
                    train_periods = periods[train_beg_idx: train_end_idx]
                    test_periods = periods[test_beg_idx: test_end_idx]

                    df_train = df.loc[df.cal_yearperiod.isin(train_periods)]
                    df_test = df.loc[df.cal_yearperiod.isin(test_periods)]

                    test_train.append((df_train, df_test))
                    # test_train[i] = {"train": df_train , "test": df_test}
      
        return test_train
    
    def showperiods(self, df, col_year = 'year' , col_period='cal_period', col_symbol='symbol'):
        df_final_periods = df.groupby(['year' , 'cal_period'])['symbol'].count().reset_index()
        df_final_periods = df_final_periods.groupby('year').agg(NoQ=('cal_period', 'count'), NoCompanies=('symbol', 'sum'))
        df_final_periods
        return df_final_periods

    def show_trainperiods(self, train_list:list, col_period):
        """ shows periods used in the windows for train and test (train,test) """

        l = []
        counter = 1
        for df_train , df_test in train_list:
            periods_train = df_train[col_period].unique().tolist()
            periods_train.sort()
            periods_test = df_test[col_period].unique().tolist()
            periods_test.sort()
            l.append((periods_train, periods_test))
            counter +=1
            print(f'window {0}: train periods= {len(periods_train)} / test periods={len(periods_test)}')

        return l


class Model(Feature):
    def __init__(self,df):
        self.df = df
        print("model object instantiated")
        
    def skpredict(self, df_train, df_test, skmodel, cols_x, cols_y, printstat=True):
        regr = skmodel
        X_train = df_train[cols_x]
        Y_train = df_train[cols_y]
        
        X_test = df_test[cols_x]
        Y_test = df_test[cols_y]
        
        regr.fit(X_train, Y_train)
        
        predict_train = regr.predict(X_train)
        predict_test = regr.predict(X_test)
        
        df_train['predict'] = predict_train
        df_train['MSE'] = (np.array(df_train[cols_y]) - predict_train)**2
        
        df_test['predict'] = predict_test
        df_test['MSE'] = (np.array(df_test[cols_y]) - predict_test)**2
        
        mse_train = metrics.mean_squared_error(Y_train, predict_train)
        mae_train = metrics.mean_absolute_error(Y_train, predict_train)
        r2_train = metrics.r2_score(Y_train, predict_train)
        
        mse_test = metrics.mean_squared_error(Y_test, predict_test)
        mae_test = metrics.mean_absolute_error(Y_test, predict_test)
        
        stat_train = {"mse": mse_train, "mae": mae_train, "r2": r2_train}
        stat_test = {"mse": mse_test, "mae": mae_test}
        
        if printstat==True:
            print(f'train stat: {stat_train}')
            print(f'test stat: {stat_test}')
            # print(f'MSE Score manually calulated: {np.mean((np.array(df_final_vf[cols_y]) - predict)**2)}')

        
        return df_train, df_test, stat_train, stat_test
    
    def skpredict_window(self, skmodel, cols_x, cols_y, col_period='cal_yearperiod' , train_window=4 , test_window=1, test_gap = 0, expanding=False):
        regr = skmodel
        train_test = self.train(self.df, col_period, train_window , test_window, test_gap, expanding)
        
        data_train = []
        data_test = []
        stat_train_times = []
        stat_test_times = []
        
        for i , j in enumerate(train_test):

            
            df_train = j[0]
            df_test = j[1]
            
            df_train, df_test, stat_train, stat_test = self.skpredict(df_train, df_test, regr, cols_x, cols_y, printstat=False)
            
            stat_train.update({"window": i, "date": df_train.cal_yearperiod.unique().tolist()[0] + "-"+ df_train.cal_yearperiod.unique().tolist()[-1]})
            stat_test.update({"window": i, "date": df_test.cal_yearperiod.unique().tolist()[0] + "-"+ df_test.cal_yearperiod.unique().tolist()[-1]})
            
            df_train['iteration'] = i
            df_test['iteration'] = i
        
            data_train.append(df_train)
            data_test.append(df_test)
            
            stat_train_times.append(stat_train)
            stat_test_times.append(stat_test)
            
            
            print(f'train stat: {stat_train}')
            print(f'test stat: {stat_test}\n')
            
        df_train_conso = pd.concat(data_train)
        df_test_conso = pd.concat(data_test)
        
        mse_train_all = df_train_conso.MSE.mean()
        mse_test_all = df_test_conso.MSE.mean()
        
        print(f'Average MSE train: {mse_train_all}')
        print(f'Average MSE test: {mse_test_all}')
        
        return df_train_conso , df_test_conso, stat_train_times, stat_test_times

def calperiod(df, col_period):
    quarter = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2", 7: "Q3", 8: "Q3",9: "Q3",10: "Q4", 11: "Q4",12: "Q4"}
    df['datetime'] = pd.to_datetime(df[col_period], format='%Y-%m-%d')
    df['yearmonth'] = df['datetime'].apply(lambda x: x.strftime('%Y%m'))
    df['year'] = df['datetime'].apply(lambda x: x.strftime('%Y'))
    df['month'] = df['datetime'].apply(lambda x: x.strftime('%m'))
    df['cal_period'] = df['month'].apply(lambda x: quarter[int(x)])
    df['cal_yearperiod'] = df['year'].astype(str) + df['cal_period']
    return df
        
    

if __name__ == '__main__':
    df_final_vf = pd.read_csv('./data/df_final_vf.csv')
    model = Model(df_final_vf)
    l_train = model.train(model.df,'cal_yearperiod', 8,1,0,True)
    
    regr = LinearRegression()
    ridge_regr = Ridge()
    xgb = ensemble.GradientBoostingRegressor()
    histgrad = ensemble.HistGradientBoostingRegressor()
    randforest = ensemble.RandomForestRegressor()
    
    regr
    
    params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error"}
    
    ### random search
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_features = ['auto', 'sqrt']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
    

    # xgb = RandomizedSearchCV(estimator = ensemble.GradientBoostingRegressor()), param_distributions = rf_grid, cv = 5, n_iter = 100)
    # search = RandomizedSearchCV(estimator = xgb, param_grid=params, scoring="f1" , cv=2)
    
    cols_y =  'usd_adjClose_t-1_change'
    cols_x = ['CASH2ASSETS_ratio_t-1_change_indavg', 'CF2CL_ratio_t-1_change_excess_indavg', 'ROA_EBITDA_ratio_indavg_t-1_change_indavg', 'usd_adjClose_t-2_change_excess_indavg' , 'usd_adjClose_t-2_change_excess_globalavg']

    # search = GridSearchCV(estimator = xgb, param_grid=rf_grid, scoring="neg_mean_squared_error")
    # search.fit(df_final_vf[cols_x], df_final_vf[cols_y])
    # search_results=search.cv_results_
    
    # df_train, df_test, stat_train, stat_test = model.skpredict(model.df, model.df, xgb, cols_x, cols_y)
    # df_train, df_test, stat_train, stat_test = model.skpredict(model.df, model.df, regr, cols_x, cols_y)
    # df_train, df_test, stat_train, stat_test = model.skpredict(model.df, model.df, ridge_regr, cols_x, cols_y)
    
    data_train_regr , data_test_regr, stat_train_regr, stat_test_regr = model.skpredict_window(regr, cols_x, cols_y, col_period='cal_yearperiod' , train_window=8 , test_window=1, test_gap = 0, expanding=True)
    
    data_train_ridge , data_test_ridge, stat_train_ridge, stat_test_ridge = model.skpredict_window(ridge_regr, cols_x, cols_y, col_period='cal_yearperiod' , train_window=8 , test_window=1, test_gap = 0, expanding=True)    
    
    data_train_xgb , data_test_xgb, stat_train_xgb, stat_test_xgb = model.skpredict_window(xgb, cols_x, cols_y, col_period='cal_yearperiod' , train_window=8 , test_window=1, test_gap = 0, expanding=True)
    
    data_train_histgrand , data_test_histgrad, stat_train_histgrand, stat_test_histgrand = model.skpredict_window(histgrad, cols_x, cols_y, col_period='cal_yearperiod' , train_window=8 , test_window=1, test_gap = 0, expanding=True)
    
    data_train_randforest, data_test_randforest, stat_train_randforest, stat_test_randforest = model.skpredict_window(randforest, cols_x, cols_y, col_period='cal_yearperiod' , train_window=8 , test_window=1, test_gap = 0, expanding=True)
    data_train_randforest['model']= 'random_forest_train'
    data_test_randforest['model']= 'random_forest_test'
    data_train_randforest.to_csv('../data/data_train_randforest.csv')
    data_test_randforest.to_csv('../data/data_test_randforest.csv')

