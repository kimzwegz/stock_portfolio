# -*- coding: utf-8 -*-

import pymongo
from pymongo import MongoClient
from data_extract import client, fin, company , companyfinancials_list, pnl, bs, cf, profile, prices, fx, mrktcap, prices_n_fin, idx_price , idx_all ,  MONGO_FIN
import pandas as pd
import datetime

db = MONGO_FIN()

df_company = db.get_df((db.company))
df_profile = db.get_df(db.profile)


df_profile2 = df_profile[[
    'Symbol' , 'companyName' , 'currency' , 'exchange',
    'exchangeShortName', 'industry' , 'sector','country', 
    'isEtf' , 'isFund' , 'isActivelyTrading']].merge(df_company[['symbol' , 'type']], how = 'left' , left_on='Symbol', right_on = 'symbol')

df_codetails = df_profile[['Symbol' , 'companyName' , 'currency', 'exchange', 'exchangeShortName', 'industry', 'sector', 'country']]
df_sample = df_codetails.loc[~(df_codetails.industry.isna()) &  ~(df_codetails.sector.isna()) &  ~(df_codetails.currency.isna()) ]

df_sample.info()


def clean_df(df, cols):
    df2 = df[cols].copy()
    for i in cols:
        df2 = df2.loc[~df2[i].isna()]
        print(i , df2.shape)
    print(df2.info())
    return df2

def q_stocks(masterlist, iterations, collection):
    l_prices = []
    for i,j in enumerate(masterlist):
        # if i <1000:
        # print (i, j['symbol'])
        date = datetime.datetime.strptime(j['date'], '%Y-%m-%d')
        res = list(collection.find({'symbol': j['symbol'] , 'date': j['date']}))
        try:
            l_prices.append(res[0])
            if i % 10000 ==0:
                print (f'{i:,} out of {len(dict_masterlist):,}')
                print(f'{len(l_prices):,} records - success {len(l_prices)/i*100:.2f} %')
        except IndexError:
            d = 1
            iteration = iter(range(iterations))
            try:
                next (iteration)
                while (d <=iterations) and (len(res)==0):
                    try:
                        # print(int(d))
                        d +=1
                        date_new = date - datetime.timedelta(days=d)
                        date_new_str = date_new.strftime('%Y-%m-%d')
                        # print(f"trying {j['symbol']} with new date {date_new_str}")
                        res = list(collection.find({'symbol': j['symbol'] , 'date': date_new_str}))
                        # print(f"{j['symbol']} found no recs at {date_new_str} iter {d+1}")
                        # print(f"found alternate data for {j['symbol']} with {date_new2_str} at iter{d}")
                        res[0]['date'] = j['date']
                        # print(f"found recs at inserting {res[0]['symbol']} with date {res[0]['date']} at iter {d}\n")
                        l_prices.append(res[0])
                    except IndexError:
                        pass
            except StopIteration:
                pass
    df = pd.DataFrame(data=l_prices)
    df['key'] = df['symbol']+"_"+df['date']
    return df

cols_co = ['date' , 'symbol','reportedCurrency', 'acceptedDate', 'fillingDate', 'calendarYear', 'period' , 'key' , 'key2']


########################## bs datacleanup ##########################

cols_bs = [
 'symbol',
 'date',
 'reportedCurrency',
 'fillingDate',
 'acceptedDate',
 'calendarYear',
 'period',
 'cashAndCashEquivalents',
 'shortTermInvestments',
 'cashAndShortTermInvestments',
 'netReceivables',
 'inventory',
 'otherCurrentAssets',
 'totalCurrentAssets',
 'propertyPlantEquipmentNet',
 'goodwill',
 'intangibleAssets',
 'goodwillAndIntangibleAssets',
 'longTermInvestments',
 'taxAssets',
 'otherNonCurrentAssets',
 'totalNonCurrentAssets',
 'otherAssets',
 'totalAssets',
 'accountPayables',
 'shortTermDebt',
 'taxPayables',
 'deferredRevenue',
 'otherCurrentLiabilities',
 'totalCurrentLiabilities',
 'longTermDebt',
 'deferredRevenueNonCurrent',
 'deferrredTaxLiabilitiesNonCurrent',
 'otherNonCurrentLiabilities',
 'totalNonCurrentLiabilities',
 'otherLiabilities',
 'totalLiabilities',
 'commonStock',
 'retainedEarnings',
 'accumulatedOtherComprehensiveIncomeLoss',
 'othertotalStockholdersEquity',
 'totalStockholdersEquity',
 'totalLiabilitiesAndStockholdersEquity',
 'totalInvestments',
 'totalDebt',
 'netDebt']
df_bs = db.get_df(db.bs)
df_bs2 = clean_df(df_bs, cols_bs)
df_bs2['key'] = df_bs2['symbol']+"_"+df_bs2['date']
df_bs2['key2'] = df_bs2['symbol']+"_"+df_bs2['date']+"_"+df_bs2['reportedCurrency']

cols_bs2 = {i: 'bs_'+i if i != "key" else i for i in df_bs2.columns}

df_bs2.rename(columns = cols_bs2 , inplace = True)
print(df_bs2.columns)

df_bs2.bs_key2.is_unique
df_bs2.key.is_unique
    
########################## pnl datacleanup ##########################

cols_pnl = [
     'date',
     'symbol',
     'reportedCurrency',
     'fillingDate',
     'acceptedDate',
     'calendarYear',
     'period',
     'revenue',
     'costOfRevenue',
     'grossProfit',
     'grossProfitRatio',
     'researchAndDevelopmentExpenses',
     'generalAndAdministrativeExpenses',
     'sellingAndMarketingExpenses',
     'sellingGeneralAndAdministrativeExpenses',
     'otherExpenses',
     'operatingExpenses',
     'costAndExpenses',
     'interestIncome',
     'interestExpense',
     'depreciationAndAmortization',
     'ebitda',
     'ebitdaratio',
     'operatingIncome',
     'operatingIncomeRatio',
     'totalOtherIncomeExpensesNet',
     'incomeBeforeTax',
     'incomeBeforeTaxRatio',
     'incomeTaxExpense',
     'netIncome',
     'netIncomeRatio',
     'eps',
     'epsdiluted',
     'weightedAverageShsOut',
     'weightedAverageShsOutDil']
df_pnl = db.get_df(db.pnl)
df_pnl2 = clean_df(df_pnl, cols_pnl)
df_pnl2['key2'] = df_pnl2['symbol']+"_"+df_pnl2['date']+"_"+df_pnl2['reportedCurrency'] 
df_pnl2['key'] = df_pnl2['symbol']+"_"+df_pnl2['date']
df_pnl2.key.is_unique
df_pnl2.drop_duplicates(subset='key', inplace=True)

cols_pnl2 = {i: 'pnl_'+i if i != "key" else i for i in df_pnl2.columns}
df_pnl2.rename(columns = cols_pnl2 , inplace = True)
print(df_pnl2.columns)
df_pnl2.pnl_key2.is_unique
df_pnl2.key.is_unique

########################## cash flow datacleanup ##########################

cols_cf = [
 'date',
 'symbol',
 'reportedCurrency',
 'fillingDate',
 'acceptedDate',
 'calendarYear',
 'period',
 'netIncome',
 'depreciationAndAmortization',
 'deferredIncomeTax',
 'stockBasedCompensation',
 'changeInWorkingCapital',
 'accountsReceivables',
 'inventory',
 'accountsPayables',
 'otherWorkingCapital',
 'otherNonCashItems',
 'netCashProvidedByOperatingActivites',
 'investmentsInPropertyPlantAndEquipment',
 'acquisitionsNet',
 'purchasesOfInvestments',
 'salesMaturitiesOfInvestments',
 'otherInvestingActivites',
 'netCashUsedForInvestingActivites',
 'debtRepayment',
 'commonStockIssued',
 'commonStockRepurchased',
 'dividendsPaid',
 'otherFinancingActivites',
 'netCashUsedProvidedByFinancingActivities',
 'effectOfForexChangesOnCash',
 'netChangeInCash',
 'cashAtEndOfPeriod',
 'cashAtBeginningOfPeriod',
 'operatingCashFlow',
 'capitalExpenditure',
 'freeCashFlow']
df_cf = db.get_df(db.cf)
df_cf2 = clean_df(df_cf, cols_cf)
df_cf2['key2'] = df_cf2['symbol']+"_"+df_cf2['date']+"_"+df_cf2['reportedCurrency']
df_cf2['key'] = df_cf2['symbol']+"_"+df_cf2['date'] 
df_cf2.key.is_unique
df_cf2.key2.is_unique

cols_cf2 = {i: 'cf_'+i if i != "key" else i for i in df_cf2.columns}
df_cf2.rename(columns = cols_cf2 , inplace = True)
print(df_cf2.columns)
df_cf2.drop_duplicates(subset='key', inplace=True)

########################## financials ##########################

df_financials = df_bs2.merge(df_pnl2, how = 'inner' , on = 'key').merge(df_cf2, how = 'inner' , on='key')
df_financials.insert(0, column = 'symbol' , value = df_financials.bs_symbol)
df_financials.insert(0, column = 'date' , value = df_financials.bs_date)    
df_financials.insert(2, "key", df_financials.pop("key"))

df_financials.drop(columns = ['bs_symbol', 'cf_symbol', 'pnl_symbol' , 'bs_date', 'cf_date', 'pnl_date'])
print(df_financials.key.is_unique)
dict_masterlist = df_financials[['symbol', 'date']].to_dict("records")

##################### market & stock data ######################

df_prices_q = q_stocks(dict_masterlist, 10,  db.prices)
df_mrktcap_q = q_stocks(dict_masterlist , 10 , db.mrktcap)

df_mrkt_q = df_prices_q.merge(df_mrktcap_q, how = 'left', on = 'key')
df_mrkt_q.insert(0 , column = 'symbol' , value = df_mrkt_q['symbol_x'])
df_mrkt_q.insert(0, column = 'key' , value = df_mrkt_q.pop('key'))
df_mrkt_q.drop(columns = ['_id_x' , '_id_y', 'symbol_x', 'symbol_y' , 'date_x' , 'date_y' , 'index'] , inplace=True)


##################### indexes ######################
df_idx_p = db.get_df((db.idx_price))
df_idx_all = db.get_df((db.idx_all))

df_idx_conso = df_idx_p.merge(df_idx_all, how = 'inner' , on = 'symbol')
df_idx_conso.to_csv('df_idx_conso.csv')



##################### consolidation ######################

df_conso1 = df_mrkt_q.merge(df_financials, how='inner' , on='key')
df_conso2 = df_conso1.merge(df_profile2, how = 'left' , left_on='symbol_x', right_on = 'Symbol')

cols_conso = [
    'key',
     'symbol',
     'date',
     'companyName',
     'currency',
     'exchange',
     'exchangeShortName',
     'industry',
     'sector',
     'country',
     'isEtf',
     'isFund',
     'isActivelyTrading',
     'symbol',
     'type',
     'date',
     'open',
     'high',
     'low',
     'close',
     'adjClose',
     'volume',
     'unadjustedVolume',
     'change',
     'changePercent',
     'vwap',
     'label',
     'changeOverTime',
     'marketCap',
     'bs_reportedCurrency',
     'bs_fillingDate',
     'bs_acceptedDate',
     'bs_calendarYear',
     'bs_period',
     'bs_cashAndCashEquivalents',
     'bs_shortTermInvestments',
     'bs_cashAndShortTermInvestments',
     'bs_netReceivables',
     'bs_inventory',
     'bs_otherCurrentAssets',
     'bs_totalCurrentAssets',
     'bs_propertyPlantEquipmentNet',
     'bs_goodwill',
     'bs_intangibleAssets',
     'bs_goodwillAndIntangibleAssets',
     'bs_longTermInvestments',
     'bs_taxAssets',
     'bs_otherNonCurrentAssets',
     'bs_totalNonCurrentAssets',
     'bs_otherAssets',
     'bs_totalAssets',
     'bs_accountPayables',
     'bs_shortTermDebt',
     'bs_taxPayables',
     'bs_deferredRevenue',
     'bs_otherCurrentLiabilities',
     'bs_totalCurrentLiabilities',
     'bs_longTermDebt',
     'bs_deferredRevenueNonCurrent',
     'bs_deferrredTaxLiabilitiesNonCurrent',
     'bs_otherNonCurrentLiabilities',
     'bs_totalNonCurrentLiabilities',
     'bs_otherLiabilities',
     'bs_totalLiabilities',
     'bs_commonStock',
     'bs_retainedEarnings',
     'bs_accumulatedOtherComprehensiveIncomeLoss',
     'bs_othertotalStockholdersEquity',
     'bs_totalStockholdersEquity',
     'bs_totalLiabilitiesAndStockholdersEquity',
     'bs_totalInvestments',
     'bs_totalDebt',
     'bs_netDebt',
     'pnl_reportedCurrency',
     'pnl_fillingDate',
     'pnl_acceptedDate',
     'pnl_calendarYear',
     'pnl_period',
     'pnl_revenue',
     'pnl_costOfRevenue',
     'pnl_grossProfit',
     'pnl_grossProfitRatio',
     'pnl_researchAndDevelopmentExpenses',
     'pnl_generalAndAdministrativeExpenses',
     'pnl_sellingAndMarketingExpenses',
     'pnl_sellingGeneralAndAdministrativeExpenses',
     'pnl_otherExpenses',
     'pnl_operatingExpenses',
     'pnl_costAndExpenses',
     'pnl_interestIncome',
     'pnl_interestExpense',
     'pnl_depreciationAndAmortization',
     'pnl_ebitda',
     'pnl_ebitdaratio',
     'pnl_operatingIncome',
     'pnl_operatingIncomeRatio',
     'pnl_totalOtherIncomeExpensesNet',
     'pnl_incomeBeforeTax',
     'pnl_incomeBeforeTaxRatio',
     'pnl_incomeTaxExpense',
     'pnl_netIncome',
     'pnl_netIncomeRatio',
     'pnl_eps',
     'pnl_epsdiluted',
     'pnl_weightedAverageShsOut',
     'pnl_weightedAverageShsOutDil',
     'cf_reportedCurrency',
     'cf_fillingDate',
     'cf_acceptedDate',
     'cf_calendarYear',
     'cf_period',
     'cf_netIncome',
     'cf_depreciationAndAmortization',
     'cf_deferredIncomeTax',
     'cf_stockBasedCompensation',
     'cf_changeInWorkingCapital',
     'cf_accountsReceivables',
     'cf_inventory',
     'cf_accountsPayables',
     'cf_otherWorkingCapital',
     'cf_otherNonCashItems',
     'cf_netCashProvidedByOperatingActivites',
     'cf_investmentsInPropertyPlantAndEquipment',
     'cf_acquisitionsNet',
     'cf_purchasesOfInvestments',
     'cf_salesMaturitiesOfInvestments',
     'cf_otherInvestingActivites',
     'cf_netCashUsedForInvestingActivites',
     'cf_debtRepayment',
     'cf_commonStockIssued',
     'cf_commonStockRepurchased',
     'cf_dividendsPaid',
     'cf_otherFinancingActivites',
     'cf_netCashUsedProvidedByFinancingActivities',
     'cf_effectOfForexChangesOnCash',
     'cf_netChangeInCash',
     'cf_cashAtEndOfPeriod',
     'cf_cashAtBeginningOfPeriod',
     'cf_operatingCashFlow',
     'cf_capitalExpenditure',
     'cf_freeCashFlow']

df_conso3 = df_conso2[cols_conso]
df_stock_mrkt_q = df_conso3.loc[df_conso3.type == 'stock']
df_stock_mrkt_q.to_csv('df_stock_mrkt_q.csv')
df_profile.to_csv('df_profile.csv')

##################### fx ######################
df_fx1 = db.get_df(fx)


curr = list(db.fx.distinct("symbol"))






