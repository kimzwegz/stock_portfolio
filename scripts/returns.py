# -*- coding: utf-8 -*-

import pymongo
from pymongo import MongoClient
from data_extract import client, fin, company , companyfinancials_list, pnl, bs, cf, profile, prices, fx, mrktcap, prices_n_fin, MONGO_FIN
import pandas as pd
import datetime


db = MONGO_FIN()

df_stocks = db.get_df(db.prices_n_fin)

df_stocks = df_stocks.loc[df_stocks.type=='stock']

df_stocks.info()
df_stocks.groupby
df_stocks.sort_values
