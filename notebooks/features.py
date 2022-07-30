# %%
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
import json

import warnings
warnings.filterwarnings('ignore')

# %%
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

# %%
## local imports

from os import pardir, path
import sys
mod_path = path.abspath(path.join(pardir))

print(mod_path)

if mod_path not in sys.path:
    sys.path.append(mod_path)

print(sys.path)
from data_extract import client, fin, company , companyfinancials_list, pnl, bs, cf, profile, prices, fx, mrktcap, prices_n_fin,  MONGO_FIN ## note needed for this notebook ignore if not using logal Mongodb
from mypackage import Feature, Model


# %%
db = MONGO_FIN()
df_csv = pd.read_csv('../data/data_vf.csv')
# df_csv = pd.read_csv('../data/data_vf_aapl.csv')
df_aapl = df_csv[(df_csv['symbol']=='AAPL') | (df_csv['symbol']=='MSFT')]
feature = Feature(df_aapl)
# feature = Feature(pd.read_csv('../data/data_vf.csv'))
df_raw = feature.df.copy()

# %%
print(df_csv.shape, df_aapl.shape)

# %%
# df_aapl = df_csv[df_csv['symbol']=='AAPL']
# df_aapl.to_csv('../data/data_vf_aapl.csv')

# %%
# df_returns_test = feature.change(df_csv, 'symbol' , 'cal_yearperiod' , 1, ['usd_adjClose'])
# df_returns_test = feature.change(df_returns_test, 'symbol' , 'cal_yearperiod' , 2, ['usd_adjClose'])
# df_returns_test = df_returns_test[['symbol', 'cal_yearperiod','usd_adjClose' ,  'usd_adjClose_t-1', 'usd_adjClose_t-1_delta' ,'usd_adjClose_t-1_change']]

# %%
df_raw.shape

# %%
def replacecol(df: pd.DataFrame, col_replace: str,replace=False):
    df2 = df.copy()
    cols = json.load(open("../config/config.json"))['cols']
    by = cols[col_replace]
    cols = df.columns.tolist()
    col_new = col_replace+"_new"
    if col_new in cols:
        df2.drop(columns=col_new,inplace=True)

    index_col = df2.columns.get_loc(col_replace)
    
    l = []
    l.append(col_replace)
    l.append(col_new)
    before = df2.loc[df2[col_replace]==0].shape[0]
    print(f'{before:,} with zeros for column {col_replace} BEFORE correction')
    if replace==False:
        # df2.insert(index_col, col_new, df2.apply(lambda row: sum([row[i] for i in by]), axis=1)) # overwrite the original data
        ## do not overwrite original data
        df2['calc'] =  df2.apply(lambda row: sum([row[i] for i in by]), axis=1)
        df2.loc[df2[col_replace]==0, col_new] = df2['calc']
        df2.loc[df2[col_replace]!=0, col_new] = df2[col_replace]
    else:
        df2['calc'] = df2.apply(lambda row: sum([row[i] for i in by]), axis=1)
        # df2[col_replace] = df2['calc']
        df2.loc[df2[col_replace]==0, col_new] = df2['calc']
        df2.loc[df2[col_replace]!=0, col_new] = df2[col_replace]
        df2[col_replace] = df2[col_new]
    
    df2['check'] = df2[col_new]- df2[col_replace]
    all_cols = l+by
    after = df2.loc[df2[col_replace]==0].shape[0]
    print(f'{after:,} with zeros for column {col_replace} AFTER correction')
    print("-----------------------------------------")
    print(f'{before-after:,} corrected for column {col_replace}\n')
    return df2[cols], df2[all_cols]

# %%
# df_appl_raw = df_raw.loc[df_raw.symbol=='AAPL']
# df_appl_raw.shape

# df_appl_raw = df_raw.loc[df_raw.symbol=='AAPL']
# df_aapl_new = replacecol(df_appl_raw, 'usd_bs_cashAndShortTermInvestments' )[1]
# df_aapl_new = replacecol(df_appl_raw, 'usd_bs_totalCurrentAssets')[1]
# df_aapl_new = replacecol(df_appl_raw, 'usd_bs_goodwillAndIntangibleAssets')[1]
# df_aapl_new = replacecol(df_appl_raw, 'usd_bs_totalNonCurrentAssets', replace=True)[0]

# %%
# df_zeroasset = df_csv.loc[(df_csv['usd_bs_totalAssets']==0) & df_csv['usd_bs_cashAndCashEquivalents']!=0]
# print(df_zeroasset.shape)

# df_zeroasset2 = replacecol(df_zeroasset, 'usd_bs_totalAssets', replace=True)[0]
# print(df_zeroasset2.shape[0])

# %%
## correct for missisng data

### correct for BS
cols_replace = [
    "usd_bs_cashAndShortTermInvestments",
    "usd_bs_totalCurrentAssets",
    "usd_bs_totalNonCurrentAssets",
    "usd_bs_totalAssets",
    "usd_bs_totalDebt",
    "usd_bs_totalNonCurrentLiabilities",
    "usd_bs_totalCurrentLiabilities",
    "usd_bs_totalLiabilities",
    "usd_bs_totalStockholdersEquity",
    "usd_pnl_sellingGeneralAndAdministrativeExpenses",
    "usd_pnl_operatingExpenses",
    "usd_pnl_costAndExpenses"
    ]

for i in cols_replace:
    df_main = replacecol(feature.df, i, replace=True)[0]

### correct for ebitda
df_main.loc[df_main['usd_pnl_ebitda']==0, 'usd_pnl_ebitda'] = df_main['usd_pnl_revenue'] - df_main['usd_pnl_costAndExpenses']


# %%

## add returns up to 4 quarters in arrears
df_returns = feature.change(df_main, 'symbol' , 'cal_yearperiod' , 1, ['usd_adjClose'])
df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 2, ['usd_adjClose'])
# df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 3, ['usd_adjClose'])
# df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 4, ['usd_adjClose'])

# df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 1, ['usd_adjClose'], log=True)
# df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 2, ['usd_adjClose'], log=True)
# df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 3, ['usd_adjClose'], log=True)
# df_returns = feature.change(df_returns, 'symbol' , 'cal_yearperiod' , 4, ['usd_adjClose'], log=True)

# %%
df_returns.shape

# %%
## add global and industry comparatives for 4 quarters returns

### simple returns averages
df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-1_change'])
df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-2_change'])

# df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-3_change'])
# df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-4_change'])

### log returns averages
# df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-1_logchange'])
# df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-2_logchange'])
# df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-3_logchange'])
# df_returns = feature.add_avg(df_returns, ['usd_adjClose_t-4_logchange'])

# %%
df_returns.shape

# %%
## check columns

df_returns.columns.tolist()

# %%
df_returns.shape

# %%
print(df_returns.columns.tolist())

# %%
# construct financial ratios as predictors

df_ratios = feature.add_ratio(df_returns,
    ## profitability ratios
    # GP=('usd_pnl_grossProfit', 'usd_pnl_revenue'),
    EBITDA=('usd_pnl_ebitda' , 'usd_pnl_revenue'),
    # OI=('usd_pnl_operatingIncome' , 'usd_pnl_revenue'),
    # NI = ('usd_pnl_incomeBeforeTax' , 'usd_pnl_revenue'),
    # ROA_OI = ('usd_pnl_operatingIncome' , 'usd_bs_totalAssets'),
    ROA_EBITDA = ('usd_pnl_ebitda' , 'usd_bs_totalAssets'),
    # ROA_NI = ('usd_pnl_incomeBeforeTax' , 'usd_bs_totalAssets'),
    # ROE_OI = ('usd_pnl_operatingIncome' , 'usd_bs_totalStockholdersEquity'),
    # ROE_EBITDA = ('usd_pnl_ebitda' , 'usd_bs_totalStockholdersEquity'),
    # ROE_NI = ('usd_pnl_incomeBeforeTax' , 'usd_bs_totalStockholdersEquity'),
    ## solvency 
    # DE=('usd_bs_totalDebt' , 'usd_bs_totalStockholdersEquity' ),
    DA = ('usd_bs_totalDebt', 'usd_bs_totalAssets'),
    CURR=('usd_bs_totalCurrentAssets' , 'usd_bs_totalCurrentLiabilities'),
    # INTCOV = ('usd_pnl_operatingIncome' , 'usd_pnl_interestExpense'),
    CASH2ASSETS = ('usd_bs_cashAndCashEquivalents' , 'usd_bs_totalAssets'),
    CF2CL=('usd_cf_netCashProvidedByOperatingActivites' , 'usd_bs_totalCurrentLiabilities'),
    CF2D=('usd_cf_netCashProvidedByOperatingActivites' , 'usd_bs_totalDebt'),
    CF2LIAB=('usd_cf_netCashProvidedByOperatingActivites' , 'usd_bs_totalLiabilities'),
    )

df_ratios.shape

# %%
cols_ratios = set(df_ratios.columns.unique())
# cols_returns = set(df_returns.columns.unique())

ratios = [i for i in cols_ratios if i[-5:] == "ratio"]
## list of all ratios created
print(ratios)

# calculate pct change of ratios vs prior period
df_ratios_change = feature.change(df_ratios, 'symbol' , 'cal_yearperiod', 1 , ratios)
print(df_ratios_change.shape)

# %%
change = list(set(df_ratios_change.columns.unique()))

cols_change = [i for i in change if i[-6:] == "change" and i[-8:]!= 'exchange' and i[:12]!= 'usd_adjClose']
print(cols_change)

# %%
df_final_v1 = feature.add_avg(df_ratios_change, cols_change)
print(df_final_v1.shape)

# %%
cols_drop = feature.cols('fin') + feature.cols('ratios') ## columns to be droped keeping featues only
df_final_v2 = df_final_v1.drop(columns=cols_drop)
df_final_v3 = df_final_v2.dropna()


# %%
df_final_v2.info(verbose=True, show_counts=True)

# %%
print(df_final_v2.shape[0], df_final_v3.shape[0])

# %%
df_final_vf = df_final_v2

# %%
df_final_periods = df_final_vf.groupby(['year' , 'cal_period'])['symbol'].count().reset_index()
df_final_periods = df_final_periods.groupby('year').agg(NoQ=('cal_period', 'count'), NoCompanies=('symbol', 'sum'))
df_final_periods

# %%
df_final_v3[df_final_v3.symbol=='AAPL'].groupby('year')['cal_yearperiod'].count()


# %%
df_aapl_vf = df_final_v2[df_final_v2.symbol=='AAPL']
df_aapl_vf

# %%
# df_final_vf = pd.read_csv('../data/df_final_vf2.csv')
# df_final_vf.to_csv('../data/df_final_vwzeros.csv')

# %%
len(df_final_vf.symbol.unique())

# %%
df_final_vf.currency


