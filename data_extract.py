

import json
import requests
from urllib.parse import urljoin
import pymongo
from pymongo import MongoClient, ASCENDING , DESCENDING
import time
import pandas as pd
import re
import json

############################### FMP Connection ################################

class FMP():
    """
    year = [integer]
    frequency = annual / quarter as string
    """
    
    # API_KEY = json.load(open("./config/api.json"))['key'] relative path
    API_KEY = json.load(open("/Users/karimkhalil/Coding/development/fin/config/api.json"))['key'] ## abs path

    URL_BULK = {
        "pnl": "https://financialmodelingprep.com/api/v4/income-statement-bulk",
        "bs": "https://financialmodelingprep.com/api/v4/balance-sheet-statement-bulk",
        "cf": "https://financialmodelingprep.com/api/v4/cash-flow-statement-bulk",
        "profile":"https://financialmodelingprep.com/api/v4/profile/all",
        "cmdt": "https://financialmodelingprep.com/api/v3/symbol/available-commodities"
        }
    
    URL = {
        "price" : "https://financialmodelingprep.com/api/v3/historical-price-full",
        "price_bulk" : "https://financialmodelingprep.com/api/v3/historical-price-full",
        "fx": 'https://financialmodelingprep.com/api/v3/historical-price-full/',
        "idx_all" :"https://financialmodelingprep.com/api/v3/symbol/available-indexes",
        "mrktcap" : "https://financialmodelingprep.com/api/v3/historical-market-capitalization",
        "indexes": "https://financialmodelingprep.com/api/v3/historical-price-full/%5E"
        }
    
    S = requests.Session()
    
    
    def __init__(self, years: list, frequency: str):
        self.years = years
        self.frequency = frequency
        
    def get_url(self, url_choice):
        if url_choice == "pnl":
            url = FMP.URL_BULK[url_choice]
        elif url_choice == "bs":
            url = FMP.URL_BULK[url_choice]
        elif url_choice == "cf":
            url = FMP.URL_BULK[url_choice]
        elif url_choice =='profile' :
            url = FMP.URL_BULK[url_choice]
        elif url_choice == "cmdt":
            url = FMP.URL_BULK[url_choice]
        elif url_choice == 'price':
            url = FMP.URL[url_choice]
        elif url_choice == 'fx':
            url = FMP.URL[url_choice]
        elif url_choice == 'indexes':
            url = FMP.URL[url_choice]
        elif url_choice == 'idx_all':
            url = FMP.URL[url_choice]
        return url
    
    def get_fx(self, fx: str, params = {}, **kwargs):
        url = urljoin(FMP.URL["fx"],fx)
        params = {'apikey' : FMP.API_KEY , 'from' : '1900-01-01'}
        params.update(kwargs)
        params["apikey"]=FMP.API_KEY
        params_print = params.copy()
        del params_print['apikey']
        with FMP.S as s:
            urls = s.request("GET" , url=url, params=params).url
            print(f'getting: {fx}')
            r = s.request("GET" , url=url, params=params).content
            loads = json.loads(r)
            symbol = loads['symbol']
            historical = loads['historical']
            for i in historical:
                i.update({'symbol': symbol})
              
        df = pd.DataFrame(historical)
        return historical, df
    
    def get_allindexes(self, params = {}):
        url = self.get_url("idx_all")
        params = {'apikey' : FMP.API_KEY}
        params_print = params.copy()
        del params_print['apikey']
        with FMP.S as s:
            urls = s.request("GET" , url=url, params=params).url
            print(f'getting: {urls}')
            r = s.request("GET" , url=url, params=params).content
            loads = json.loads(r)
        return loads
        
    def get_index (self, tickers: list , params = {}):
        l =[]
        l_notfound = []
        counter = 0
        no_tickers = len(tickers)
        print(f'no of tickers: {no_tickers:,}')
        for ticker in tickers:
            url = FMP.URL["indexes"]+ticker         
            params["apikey"]=FMP.API_KEY
            params_print = params.copy()
            del params_print['apikey']
            # print(url, params)
            with FMP.S as s:
                urls = s.request("GET" , url=url, params=params).url
                print(urls)
                r = s.request("GET" , url=url, params=params).content
                loads = json.loads(r)
                if len(loads)==0:
                    print(f'no records found for {ticker} with {params_print}')
                    l_notfound.append(ticker)
                else:
                    # print(len(loads))
                    # return loads
                    ticker = re.sub(r'[^\w]', "", loads['symbol'])
                    history = loads['historical']
            # return history , ticker
                    symbol = {'symbol': ticker}
                    for i in history:
                        i.update(symbol)
                    df = pd.DataFrame(history)
                    print(f'found {df.shape[0]:,} rows for {ticker}')
                    l.append(df)
                    counter +=1
            print(f'{counter} out of {no_tickers} tickers \n')
        try:
            df2 = pd.concat(l)
            df2.reset_index(inplace=True)
            print(f"dataframe complete. {df2.shape[0]:,} rows - {df2.shape[1]} columns")
            df_dict = df2.to_dict("records")
            print(f'mongo data ready. {len(df_dict):,} records \n')
            return df2 , df_dict, l_notfound
        except ValueError:
            print ('no records to concatenate')
    
    def mrktcap (self, tickers: list, params = {}):
        l =[]
        l_notfound = []
        counter = 0
        no_tickers = len(tickers)
        print(f'no of tickers: {no_tickers:,}')
        for ticker in tickers:
            url = FMP.URL["mrktcap"]+"/"+ticker         
            params["apikey"]=FMP.API_KEY
            params_print = params.copy()
            del params_print['apikey']
            with FMP.S as s:
                urls = s.request("GET" , url=url, params=params).url
                # print(urls)
                r = s.request("GET" , url=url, params=params).content
                loads = json.loads(r)
                if len(loads)==0:
                    print(f'no records found for {ticker} with {params_print}')
                    l_notfound.append(ticker)
                else:

                    history = [{"symbol" : i['symbol'] , 'date': i['date'] , 'marketCap': i['marketCap']} for i in loads]

                    df = pd.DataFrame(history)
                    print(f'found {df.shape[0]:,} rows for {ticker}')
                    l.append(df)
                    counter +=1
        try:
            df2 = pd.concat(l)
        #     df2.reset_index(inplace=True)
            print(f"dataframe complete. {df2.shape[0]:,} rows - {df2.shape[1]} columns")
            df_dict = df2.to_dict("records")
            print(f'mongo data ready. {len(df_dict):,} records \n')
            return df2 , df_dict, l_notfound
        except ValueError:
            return ('no records to concatenate')
    
    def insertdb_mrktcap(self, db , batch_size, tickers: list):
        dicts = []
        finished_t = []
        l_notfound = []
        l_found = []
        counter = 0
        records = 0
        no_tickers = len(tickers)
        print(f'procedure to insert: {no_tickers:,} tickers')
        beg=time.time()
        start=time.time()
        for ticker in tickers:
            counter +=1
            print(f'processing {counter} out of {no_tickers}')
            try:
                ticker = [ticker]
                df_price , dict_price , l = self.mrktcap(ticker)
                dicts += dict_price
                l_found.append(ticker)
                completed = len(l_found)
                if completed % batch_size == 0 or no_tickers == completed + len(l_notfound):
                    db.    ny(dicts)
                    records += len(dicts)
                    dicts =[]
                    print("###########################################################")
                    print(f'{completed} inserted out of {no_tickers}')
                    end = time.time()
                    sec = end-start
                    print(f'{records} records inserted in {sec:.2f} seconds {sec/60:.2f} minutes elapsed')
                    print("########################################################### \n")
            except Exception as e:
                print(f'{e} for {ticker} \n')
                l_notfound += ticker
        finish = time.time()
        totalsec = finish-beg
        totalmin = totalsec/60
        print("###########################################################")
        print("###########################################################")
        print("######################### Summary $########################")
        print(f'inserted {len(l_found)} tickers')
        print(f"total {len(db.distinct('symbol')):,} tickers in db")
        print(f'not found {len(l_notfound)} tickers')
        print(f'total request for {len(tickers)}')
        if len(l_found) + len(l_notfound) == len(tickers):
            print('total found and not found matches total tickers \n')
        else:
            print(f'mismatch of found and not found with number of tickers: {len(l_found) + len(l_notfound) - len(tickers)}')
        print(f'total records {records:,} inserted')
        print(f'total records {db.count_documents({}):,} in db')
        if records==db.count_documents({}):
            print('total records inserted matches records in db :)')
        else:
            print('mismatch in record inserted and records in db')
        print(f'{totalsec:.2f} seconds : {totalmin:.2f} minutes elapsed')
        print("###########################################################")
        print("###########################################################")
        return l_found , l_notfound
              
    def session_bulk(self, url_choice: str, params = {} , **kwargs):
        url = self.get_url(url_choice)
        print(url)
        params.update(kwargs)
        params["apikey"]=FMP.API_KEY
        params_print = params.copy()
        del params_print['apikey']
        # print(params , params_print)
        with FMP.S as s:
            urls = s.request("GET" , url=url, params=params).url
            print(urls)
            try:
                r = s.request("GET" , url=url, params=params).content    
                loads = json.loads(r)
                print(loads)
                df = pd.read_json(urls)
            except ValueError:
                df = pd.read_csv(urls)
                if df.shape[1] == 1:
                    print(f'no data in found for {params_print}')
                    pass
                else:
                    print(f'found {df.shape[0]:,} rows : {df.shape[1]} columns for {params_print}')
            return df
        
    def price_tseries(self, url_choice, tickers: list, params = {}):
        url = self.get_url(url_choice)
        l =[]
        l_notfound = []
        counter = 0
        no_tickers = len(tickers)
        print(f'no of tickers: {no_tickers:,}')
        for ticker in tickers:
            url = FMP.URL["price"]+"/"+ticker         
            params["apikey"]=FMP.API_KEY
            params_print = params.copy()
            del params_print['apikey']
            # print(url, params)
            with FMP.S as s:
                urls = s.request("GET" , url=url, params=params).url
                # print(urls)
                r = s.request("GET" , url=url, params=params).content
                loads = json.loads(r)
                if len(loads)==0:
                    print(f'no records found for {ticker} with {params_print}')
                    l_notfound.append(ticker)
                else:
                    # print(len(loads))
                    # return loads
                    ticker = loads['symbol']
                    history = loads['historical']
                    symbol = {'symbol': ticker}
                    
        # return history
                    for i in history:
                        i.update(symbol)
                        # del i['index']
                    df = pd.DataFrame(history)
                    print(f'found {df.shape[0]:,} rows for {ticker}')
                    l.append(df)
                    counter +=1
                    # main.update(history)
            # print(f'{counter} out of {no_tickers} tickers \n')
        try:
            df2 = pd.concat(l)
            df2.reset_index(inplace=True)
            print(f"dataframe complete. {df2.shape[0]:,} rows - {df2.shape[1]} columns")
            df_dict = df2.to_dict("records")
            print(f'mongo data ready. {len(df_dict):,} records \n')
            return df2 , df_dict, l_notfound
        except ValueError:
            return ('no records to concatenate')
        
        
        
            
    def stock_price (self, tickers: list , timeS:int , params = {}):
        l =[]
        l_notfound = []
        counter = 0
        no_tickers = len(tickers)
        print(f'no of tickers: {no_tickers:,}')
        for ticker in tickers:
            url = FMP.URL["price"]+"/"+ticker         
            params.update({"timeseries": timeS})
            params["apikey"]=FMP.API_KEY
            params_print = params.copy()
            del params_print['apikey']
            # print(url, params)
            with FMP.S as s:
                urls = s.request("GET" , url=url, params=params).url
                # print(urls)
                r = s.request("GET" , url=url, params=params).content
                loads = json.loads(r)
                if len(loads)==0:
                    print(f'no records found for {ticker} with {params_print}')
                    l_notfound.append(ticker)
                else:
                    # print(len(loads))
                    # return loads
                    ticker = loads['symbol']
                    history = loads['historical']
                    symbol = {'symbol': ticker}
                    
        # return history
                    for i in history:
                        i.update(symbol)
                        # del i['index']
                    df = pd.DataFrame(history)
                    print(f'found {df.shape[0]:,} rows for {ticker}')
                    l.append(df)
                    counter +=1
                    # main.update(history)
            # print(f'{counter} out of {no_tickers} tickers \n')
        try:
            df2 = pd.concat(l)
            df2.reset_index(inplace=True)
            print(f"dataframe complete. {df2.shape[0]:,} rows - {df2.shape[1]} columns")
            df_dict = df2.to_dict("records")
            print(f'mongo data ready. {len(df_dict):,} records \n')
            return df2 , df_dict, l_notfound
        except ValueError:
            return ('no records to concatenate')
    
    def insertdb_price(self, db, batch_size, tickers: list , timeS:int):
        dicts = []
        finished_t = []
        l_notfound = []
        l_found = []
        counter = 0
        records = 0
        no_tickers = len(tickers)
        print(f'procedure to insert: {no_tickers:,} tickers')
        beg=time.time()
        start=time.time()
        for ticker in tickers:
            counter +=1
            print(f'processing {counter} out of {no_tickers} ==> completed: {len(l_found)}')
            try:
                ticker = [ticker]
                df_price , dict_price , l = self.stock_price(ticker,timeS)
                dicts += dict_price
                l_found.append(ticker)
                completed = len(l_found)
                if counter % batch_size == 0 or no_tickers == completed + len(l_notfound):
                    db.insert_many(dicts)
                    records += len(dicts)
                    dicts =[]
                    print("###########################################################")
                    print(f'{completed} inserted out of {no_tickers}')
                    end = time.time()
                    sec = end-start
                    print(f'{records} records inserted in {sec:.2f} seconds {sec/60:.2f} minutes elapsed')
                    print("########################################################### \n")
            except Exception as e:
                print(f'{e} for {ticker} \n')
                l_notfound += ticker
        finish = time.time()
        totalsec = finish-beg
        totalmin = totalsec/60
        print("###########################################################")
        print("###########################################################")
        print("######################### Summary $########################")
        print(f'inserted {len(l_found)} tickers')
        print(f"total {len(db.distinct('symbol')):,} tickers in db")
        print(f'not found {len(l_notfound)} tickers')
        print(f'total request for {len(tickers)}')
        if len(l_found) + len(l_notfound) == len(tickers):
            print('total found and not found matches total tickers \n')
        else:
            print(f'mismatch of found and not found with number of tickers: {len(l_found) + len(l_notfound) - len(tickers)}')
        print(f'total records {records:,} inserted')
        print(f'total records {db.count_documents({}):,} in db')
        if records==db.count_documents({}):
            print('total records inserted matches records in db :)')
        else:
            print('mismatch in record inserted and records in db')
        print(f'{totalsec:.2f} seconds : {totalmin:.2f} minutes elapsed')
        print("###########################################################")
        print("###########################################################")
        return l_found , l_notfound
        
    
    def stock_price_bulk(self, tickers: list , params = {}):
        tickers = ",".join(tickers)
        print(tickers)
        url = FMP.URL["price"]+"/"+tickers       
        params["apikey"]=FMP.API_KEY
        params_print = params.copy()
        del params_print['apikey']
        print(url, params)
        with FMP.S as s:
            urls = s.request("GET" , url=url, params=params).url
            print(urls)
            r = s.request("GET" , url=url, params=params).content
            loads = json.loads(r)
            if len(loads)==0:
                print(f'no records found for {tickers}')
            else:
                result = loads['historicalStockList']
                l = []
                for i in result:
                    symbol = i['symbol']
                    df = pd.DataFrame(i['historical'])
                    df.insert(0,column='symbol',value='symbol')
                    l.append(df)
                return l
                
    def get_bulk_timeS(self, url_choice):
        l = []
        for i in self.years:
            df = self.session_bulk(url_choice, year=i , period=self.frequency)
            l.append(df)
        print("\n")
        print("compiling dataframe")
        df2 = pd.concat(l)
        df2.reset_index(inplace=True)
        print(f"dataframe complete. {df2.shape[0]:,} rows - {df2.shape[1]} columns")
        df_dict = df2.to_dict("records")
        print(f'mongo data ready. {len(df_dict):,} records')
        return df2 , df_dict
  
class MONGO_FIN():
    __CLIENT = MongoClient('localhost' , 27017)
    __FIN = __CLIENT.db_fin
    def __init__(self):
        self.company = MONGO_FIN.__FIN.company_today
        self.companyfinancials_list = MONGO_FIN.__FIN.company_wfinancials
        self.profile = MONGO_FIN.__FIN.profile
        self.cmdt_all = MONGO_FIN.__FIN.cmdt_all
        self.cmdt_prices = MONGO_FIN.__FIN.cmdt_prices
        self.pnl = MONGO_FIN.__FIN.pnl
        self.bs = MONGO_FIN.__FIN.bs
        self.cf = MONGO_FIN.__FIN.cf
        self.prices = MONGO_FIN.__FIN.prices
        self.mrktcap = MONGO_FIN.__FIN.mrktcap
        self.fx = MONGO_FIN.__FIN.fx
        self.prices_n_fin = MONGO_FIN.__FIN.prices_n_fin
        self.idx_price = MONGO_FIN.__FIN.idx_price
        self.idx_all = MONGO_FIN.__FIN.idx_all
        self.mldf = MONGO_FIN.__FIN.mldf
        self.mlstat = MONGO_FIN.__FIN.mlstat
        
    def get_df(self, table, query: dict = None):
        df = pd.DataFrame(list(table.find()))
        return df
                       
############################# Mongodb Connection ##############################        
client = MongoClient('localhost' , 27017)
fin = client.db_fin
print(client.list_database_names())
print(fin.list_collection_names())
company = fin.company_today
companyfinancials_list = fin.company_wfinancials
pnl = fin.pnl
bs = fin.bs
cf = fin.cf
profile = fin.profile
cmdt_all = fin.cmdt_all
cmdt_prices = fin.cmtd_prices
prices = fin.prices
fx = fin.fx
mrktcap = fin.mrktcap
prices_n_fin = fin.prices_n_fin
idx_all = fin.idx_all
idx_price = fin.prices_indexes
mldf = fin.mldf
mlstat = fin.mlstat
# len(prices.distinct("symbol"))

fin.prices.create_index([("symbol" , ASCENDING),("date" , ASCENDING)])
fin.fx.create_index([("symbol" , ASCENDING),
                     ("date" , ASCENDING),
                     ("label" , ASCENDING)])
fin.mrktcap.create_index([("symbol" , ASCENDING),("date" , ASCENDING)])

fin.prices_n_fin.create_index([("symbol" , ASCENDING),
                               ("date" , ASCENDING) , 
                               ("type" , ASCENDING),
                               ("datetime" , ASCENDING),
                               ("month" , ASCENDING),
                               ("year" , ASCENDING)])

fin.idx_price.create_index([("symbol" , ASCENDING),
                            ("date" , ASCENDING)])

fin.cmdt_prices.create_index([("symbol" , ASCENDING),
                            ("date" , ASCENDING)])

############################# Instance Creation ###############################

if __name__ == "__main__":
    years = [1960 +i for i in range(63)]
    fmp = FMP(years , "quarter")
    df_profile = fmp.session_bulk('profile')
    symbols_fmp = df_profile.symbol.unique().tolist()
     
    
############################## commodities ####################################

    df_cmdt = fmp.session_bulk('cmdt')
    cmdt_l = df_cmdt.symbol.unique().tolist()
    df_cmdt_prices = fmp.price_tseries('cmdt', cmdt_l, {"timeseries": 50*365})[0]
    # fin.cmdt_all.insert_many(df_cmdt.to_dict('records'))
    # fin.cmdt_prices.insert_many(df_cmdt_prices.to_dict('records'))
          
############################### Mrktcap ##################################

    symbols = pd.DataFrame(list(companyfinancials_list.find()))['symbol'].tolist()
    # found , notfound = fmp.insertdb_mrktcap(fin.mrktcap, 1000, symbols)

############################### currency ######################################
    currencies = fin.bs.find().distinct("reportedCurrency")
    currencies = [i+'USD' for i in currencies if isinstance(i, str)]
    
    
    l_df_currencies = []
    l_dict_currencies = []
    for i in currencies:
        try:
            l_df_currencies.append(fmp.get_fx(i)[1])
            for j in fmp.get_fx(i)[0]:
                l_dict_currencies.append(j)
        except Exception as e:
            print(f'{e}: {i} not found')
            
    
    fin.fx.insert_many(l_dict_currencies)
    
    
################################## Tests ######################################
    # test = ['AAPL' , 'MSFT']
    # loads = fmp.stock_price_bulk(['AAPL' , 'MSFT'])
    # AAPL = fmp.stock_price(['AAPL'], 365*30)
    # sym = symbols[:35]
    # df_prices , dict_prices, l = fmp.stock_price(['AAPL'] , 365 * 40)

    
############################### Stock Prices ##################################
    df_prices, dict_prices = fmp.stock_price(symbols, 365 * 40)
    df_cf , dict_cf = fmp.get_bulk_timeS('cf')
    df_bs , dict_bs = fmp.get_bulk_timeS('bs')
    df_pnl , dict_pnl = fmp.get_bulk_timeS('pnl')
    finished = fin.prices.distinct("symbol")
    finished = set(finished)
    all_tickers = set(symbols)
    sym_remain = all_tickers.difference(finished)
    # found , notfound = fmp.insertdb_price(fin.prices, 1000, sym_remain , 365 * 40)
    found , notfound = fmp.insertdb_price(fin.prices, 1000, all_tickers , 365 * 40)
    # found , notfound = fmp.insertdb_price(fin.prices, 1000, ['AAPL' , 'MSFT'] , 365 * 40)
    

############################### indexes Prices ################################
    all_indexes = fmp.get_allindexes()
    for i in all_indexes:
        i.update({'symbol' :i['symbol'][1:]})
    
    
    
    
    l_allindexes = [re.sub(r'[^\w]', "", i['symbol']) for i in all_indexes]
    df_prices_index , dict_prices_index , l_prices_index_notfound = fmp.get_index(l_allindexes, {'from' : '1900-01-01'})
    # fin.prices_indexes.insert_many(dict_prices_index)
    
    df_prices_index['datetime'] = pd.to_datetime(df_prices_index.date , format="%Y-%m-%d")
    df_prices_index['month'] = df_prices_index.datetime.dt.month
    df_prices_index['year'] = df_prices_index.datetime.dt.year
    fin.idx_price.insert_many(df_prices_index.to_dict('records'))
    fin.idx_all.insert_many(all_indexes)
    
############################### insert into DB ################################
# if __name__ == "__main__":
#     cf.insert_many(dict_cf)
#     prices.insert_many(dict_prices)