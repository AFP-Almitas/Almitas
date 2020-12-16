#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:01:43 2020
@author: kanp
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from cefpanelreg0 import CEFpanelreg
from datetime import datetime, date, timedelta
import statsmodels.api as sm
"""
STEP1: Input file name
"""

filename = 'US_data_2.csv'
filename2 = 'US_data.csv'
filename3 = 'Treasury.csv'
"""
STEP2: Input parameters for regression
- start datetime, end datetime in 'YYYY-MM-DD'
- y: what you want to predict; default is 'cd'
- var_pit: Point-in-time independent variables in [variable, lag]; unit in day
    e.g. ['volume',1] >> regress on lag1 of volume2q y765reszvhui98ytgfc ml[po;kjn  但但他妳他有他有他有但這這有在但在在他]
- var_norm: Normalized independent variables in [variable, lag, length, func]; unit in day
    e.g. [cd,1,3,mean] >> regress on 3-day mean from lag1 of cd
- fix: Fixed effects; choose one from ['assetclasslevel1','assetclasslevel2','assetclasslevel3']
- Cluster: Covariance clustering; choose from ['year','ticker']
"""
file = pd.read_csv(filename, index_col = 0)
daily_file = pd.read_csv(filename2)
treasury = pd.read_csv(filename3)


cef = CEFpanelreg(file)
cef.result(
        start_datetime = '1999-01-04',
        end_datetime = '2015-12-31',
        y = ['cd5'],
        var_pit = [['pd',5], ['navchg',5]],
        var_norm = [['pd',5,10,'mean']],
        fix = ['assetclasslevel2'],
        cluster = ['year','ticker']
        )
cef.summary()


# backtest
train_asset = pd.Series(cef.assetclass.iloc[:,0]).sort_values().reset_index(drop=True)
# drop first asset (the intercept)
coef_asset = train_asset[1:].reset_index(drop=True)
    
# filter dates
start_datetime = '2015-12-20'
end_datetime = '2019-12-31'
validation = cef.data.loc[(cef.data['date']>=start_datetime) & (cef.data['date']<=end_datetime)]

# filter columns
y = ['cd5']
fix = ['assetclasslevel3']
validation = validation[y + ['year','ticker'] + [col for col in validation.columns[cef.c:]] + fix + ['date', 'ret', 'day','priceclose']]
validation = validation.dropna()
validation = validation.set_index(['ticker','year'])
asset = pd.Series(validation.assetclasslevel3.unique()).sort_values().reset_index(drop=True)

# create alldays and allrpices to temporarilly store day and prices
alldays = validation['day']
alldays = alldays.reset_index()
alldays = alldays['day']
allprices = validation['priceclose']
allprices = allprices.reset_index()
allprices = allprices['priceclose']

# drop columns 'day' and 'priceclose' to fit the number of columns 
validation = validation.drop(columns=['day','priceclose'])

# extract coefficients from fitted model
fit = cef.result
coef = fit._params

# check if asset classes in validation set is also in the training data set
check = np.zeros((len(asset)))
for i in range(0, len(asset)) :
    check[i] = any(train_asset.str.contains(asset[i]))

# construct matrix of independent variables
# start with array of ones for the intercept
intercept = np.mat(np.repeat(1,len(validation)))

# manually create columns of 1 for each asset class
fix_asfactors = pd.DataFrame(np.zeros((len(validation), len(coef_asset))))
assetclasscol = validation[fix].reset_index(drop=True)
fix_asfactors = pd.concat([fix_asfactors, assetclasscol], axis=1)

for i in range(0, len(coef_asset)) : 
    index = fix_asfactors[fix]==coef_asset[i]
    index = index.iloc[:,0] 
    fix_asfactors.loc[index,i] = 1

# drop assetclasslevel3 column as it is no longer used
fix_asfactors = fix_asfactors.drop(columns=fix)

# select columns of independent variables
indeptvar = validation.iloc[:,1:-3]

# construct the matrix of variables
x = np.append(intercept.T,fix_asfactors, axis=1)
x = np.append(x, indeptvar, axis=1)

# predict cd based on coefficients from model fitted on training data set
pred = pd.DataFrame(np.matmul(x, coef).T)

# merge prediction with validation data set
validation = validation.reset_index()
validation = pd.concat([validation, pred], axis=1)
validation= validation.rename({0: 'cdpred'}, axis='columns')


# include day and prices 
validation['day'] = alldays
validation['priceclose'] = allprices

# subset validation as only Tuesday data
validation2 = validation.loc[validation['day'] == 'Tuesday'] 

# calculate weekly returns for each CEF
validation2['ret'] = validation2.groupby('ticker').priceclose.pct_change().fillna(0)
validation2 = validation2.loc[validation.date >= datetime(2016, 1, 1)]

# form portfolios and compute weekly returns
date = validation2['date'].unique()
port = pd.DataFrame(np.zeros((len(date), 2)), columns=['longonly', 'longshort'])

# unify the date types 
treasurydate = [datetime.strptime(date, '%Y-%m-%d').date() for date in treasury['DATE']]
treasurydatetime = [datetime(my_date.year, my_date.month, my_date.day) for my_date in treasurydate] 
date_datetime = [datetime.utcfromtimestamp(temp_date.tolist()/1e9) for temp_date in date]

# subset the treasury with the same dates as our data
treasury['DATE'] = treasurydatetime
treasury_period = treasury.loc[treasury['DATE'].isin(date)]
treasury_period = treasury_period.reset_index()

# daily data for calculating average daily volume
start_datetime2 = '2015-07-01'
end_datetime2 = '2020-12-31'
daily_CEF = daily_file.loc[(daily_file['date']>=start_datetime2) & (daily_file['date']<=end_datetime2)]
daily_CEF = daily_CEF.reset_index()
daily_CEF['date'] = [datetime.strptime(dd,'%Y-%m-%d') for dd in daily_CEF['date']]
#daily_CEF['ticker'] = [tk.split()[0] for tk in daily_CEF['ticker']]


daily_avg_vol_df = pd.DataFrame()
    
# calculating daily average volume
for i in range(0, len(date)):
    days_before = date[i] - pd.to_timedelta(60, unit = 'd')
    aaa = datetime.utcfromtimestamp(date[i].tolist()/1e9)
    days_before2 = days_before.to_pydatetime()
    
    test = daily_CEF.loc[(daily_CEF['date'] < aaa) & (daily_CEF['date'] >= days_before2)]
    
    avg_vol_df = test[['ticker', 'volume']].groupby('ticker').mean()
    avg_vol_df['date'] = np.repeat(date[i], len(avg_vol_df['volume']))
    avg_vol_df = avg_vol_df.reset_index()
    
    avg_vol_df.columns = ['ticker', 'avg_daily_vol', 'date']
    
    daily_avg_vol_df = daily_avg_vol_df.append(avg_vol_df)
'''
daily_avg_vol= daily_avg_vol_df[['ticker', 'avg_daily_vol', 'date']]
daily_avg_vol_df = daily_avg_vol_df.reset_index()
daily_avg_vol_df['']
'''
validation2['next_ret'] = validation2.groupby('ticker').ret.shift(-1).fillna(0)
validation2['next_cdpred'] = validation2.groupby('ticker').cdpred.shift(-1).fillna(0)

  
validation2 = validation2.merge(daily_avg_vol_df, on = ['ticker', 'date'], how = 'left')


# load S&P500 data to compute CEF beta
# merge with S&P and Bond data
rawdata = pd.read_excel(r'S&Pdata.xlsx', skiprows=6, sheet_name=[0,1])
# merge Barclays Bond index and S&P500 data
benchmark = rawdata[0].iloc[:,0:3].merge(rawdata[1].iloc[:,0:3], on='Name')
benchmark.columns = ['Date', 'Bond Price', 'Bond MV', 'S&P Price', 'S&P MV']

# compute returns 
benchmark['lag Bond Price'] = benchmark['Bond Price'].shift(1)
benchmark['Bond ret'] = benchmark['Bond Price']/benchmark['lag Bond Price'] - 1
benchmark['lag S&P Price'] = benchmark['S&P Price'].shift(1)
benchmark['S&P ret'] = benchmark['S&P Price']/benchmark['lag S&P Price'] - 1


begin_cap = 1000000
bidask = 0.01
commission = 0.004
file['outstanding'] = file['marketcap.x']/file['priceclose']

strat_beta = []
cap = [] 
cap.append(begin_cap)

shares = pd.DataFrame()

for t in range(0, len(date)-1) : 
    # sort CEFs into decile in each week, based on cdpred
    print(t)
    dt = validation2.loc[validation2['date']==date[t]]

    file['date'] = pd.to_datetime(file['date'])

    dt2 = dt.merge(file[['ticker','date','outstanding']], on = ['ticker','date'], how = 'left')
    
    # form long only EW portfolio from the top decile
    
    dt2['decile'] = pd.qcut(dt2['next_cdpred'], 10, labels=False)
    
    top = dt2.loc[dt2.decile==9]
    bot = dt2.loc[dt2.decile==0]
    #port.loc[t, 'longonly'] = np.mean(dt2.loc[dt2.decile==9]['ret'])
    
    # form long-short EW portfolio from the top and bottom decile
    #port.loc[t, 'longshort'] = np.mean(dt2.loc[dt2.decile==9]['ret']) - 
     #   np.mean(dt2.loc[dt2.decile==0]['ret'])
    cap_each = cap[t]/len(top.index)
    
    top['ew_shares'] = cap_each/top['priceclose']
    top['ew_shares'] = np.floor(np.array(top[['ew_shares']]))
    top['vol_limit'] = top['avg_daily_vol']*0.2
    top['vol_limit'] = np.nan_to_num(top['vol_limit'], nan = cap[t])
    top['outstanding_limit'] = top['outstanding']*0.03
    top['div_limit_money'] = np.repeat(cap[t]/20, len(top['ticker']))
    top['div_limit_shares'] = np.floor(np.array(top['div_limit_money']/top['priceclose']))
    
    top['shares_traded'] = np.minimum(top['ew_shares'],top['vol_limit'])
    top['shares_traded'] = np.minimum(top['shares_traded'],top['div_limit_shares'])
    top['shares_traded'] = np.minimum(top['shares_traded'],top['outstanding_limit'])
    
    top['money_invested'] = top['shares_traded']*top['priceclose']
    

    top['money_ret'] = top['money_invested'] * (1+top['next_ret'])
    
    shares_td = top[['ticker','date','shares_traded']]
    
    if t == 0:
        shares_ytd = shares_td
        shares_ytd['shares_traded'] = np.repeat(0, len(shares_td['shares_traded']))

    compare_shares = shares_td.merge(shares_ytd, on = ['ticker'], how = 'outer').fillna(0)
    compare_shares['buy'] = np.maximum(compare_shares['shares_traded_x'] - compare_shares['shares_traded_y'],
                                       np.repeat(0, len(compare_shares['shares_traded_x'])))
    compare_shares['sell'] = np.maximum(compare_shares['shares_traded_y'] - compare_shares['shares_traded_x'],
                                       np.repeat(0, len(compare_shares['shares_traded_x'])))
    #shares = shares.append(top[['ticker','date','shares_traded']])
    
    total_shr_buy = sum(compare_shares['buy'])
    total_shr_sell = sum(compare_shares['sell'])
    total_shr = total_shr_buy + total_shr_sell
    money_in_CEF = sum(top['money_invested'])
    money_in_treasury = cap[t] - money_in_CEF 
    
    bidask_fee = total_shr*bidask/2
    commission_fee = total_shr*commission
    tracost = bidask_fee + commission_fee
    
    tot_ret = sum(top['money_ret']) + money_in_treasury*(1+ (float(treasury_period['DGS1MO'][t])/100)/52)

    ret_w_cost = tot_ret - tracost
    
    cap.append(ret_w_cost)
    shares_ytd = shares_td 
    
    # compute strategy beta
    top['actual_weight'] = top['money_invested']/sum(top['money_invested'])
    # select ticker in the portfolio
    ticker = top.ticker.unique()
    for i in range(0,len(ticker)) :
        # select past return data for the relevant ticker
        ret_data = file.loc[file.ticker==ticker[i]][['ticker', 'date', 'daily_rtn']]
        ret_data = ret_data.loc[(ret_data.date<date[t]) & (ret_data.date>=benchmark.Date.min())]
        
        # use past 1 year data to estimate beta
        ret_data = ret_data.iloc[max(0,len(ret_data)-252):len(ret_data)]
        ret_data = ret_data.dropna()
        
        # regress cef return on S&P return
        y = ret_data.daily_rtn.reset_index(drop=True)
        x = benchmark.loc[benchmark.Date.isin(ret_data.date), 'S&P ret'].reset_index(drop=True)
        x = sm.add_constant(x)
    
        mod = sm.OLS(y, x)
        res = mod.fit()
        top.loc[top.ticker==ticker[i],'beta'] = res.params[1]
        
    # compute overall portfolio beta at time t
    strat_beta.append(sum(top['actual_weight']*top['beta']))
    

cap = np.array(cap)
cap_shift = np.roll(cap, 1)

week_ret = cap/cap_shift - 1
week_ret[0] = 0
week_ret = np.delete(week_ret,0)

rf = [float(rf_rate)/100/52 for rf_rate in treasury_period['DGS1MO']]
rf = rf[:-1]
excess_ret = week_ret-rf

cum_ret = cap/1000000 - 1
cum_ret = np.delete(cum_ret,0)
cum_ex_ret = np.cumprod(1+excess_ret)-1
print(cum_ex_ret)

ret_vol = np.std(week_ret)
mean_ret = np.mean(excess_ret)

Sharpe = (mean_ret/ret_vol)*(52**0.5)
print(Sharpe)

# compute benchmark returns
# merge with S&P and Bond data
rawdata = pd.read_excel(r'S&Pdata.xlsx', skiprows=6, sheet_name=[0,1])
# merge Barclays Bond index and S&P500 data
benchmark = rawdata[0].iloc[:,0:3].merge(rawdata[1].iloc[:,0:3], on='Name')
benchmark.columns = ['Date', 'Bond Price', 'Bond MV', 'S&P Price', 'S&P MV']
# subset data on relevant dates
benchmark = benchmark.loc[benchmark['Date'].isin(date)]
benchmark = benchmark.reset_index()
# compute returns 
benchmark['lag Bond Price'] = benchmark['Bond Price'].shift(1)
benchmark['Bond ret'] = benchmark['Bond Price']/benchmark['lag Bond Price'] - 1
benchmark['lag S&P Price'] = benchmark['S&P Price'].shift(1)
benchmark['S&P ret'] = benchmark['S&P Price']/benchmark['lag S&P Price'] - 1
benchmark['60/40 ret'] = 0.6*benchmark['S&P ret'] + 0.4*benchmark['Bond ret']

benchmark['60/40 cumret'] = np.cumprod(1+benchmark['60/40 ret']) - 1


date2 = np.delete(date,0)
plt.figure(figsize=(10, 6))
plt.plot(date2, cum_ret, label='Long-Only EW Portfolio')
plt.plot(date2, benchmark.loc[1:,'60/40 cumret'], label='60-40 Portfolio')
plt.ylabel('Cumulative Return')
plt.title('Performance of Portfolios Constructed with Model on Weekly Frequency Data')
plt.legend(loc='lower right')


# plot beta of strategy
date2 = np.delete(date,0)
plt.figure(figsize=(10, 6))
plt.plot(date2, strat_beta, label='Portfolio Beta')
plt.ylabel('Beta')
plt.title('Portfolio Beta across Time')
plt.legend(loc='lower right')

print(np.mean(strat_beta))

# compute hedged return of the strategy
beta_hedge = strat_beta*benchmark.loc[1:,'S&P ret']
hedged_ret = week_ret - beta_hedge
hedged_excess_ret = hedged_ret-rf

cum_hedged_ret = np.cumprod(1+hedged_ret)-1

hedged_ret_vol = np.std(hedged_ret)
mean_hedged_ret = np.mean(hedged_excess_ret)

Sharpe_hedged = (mean_hedged_ret/hedged_ret_vol)*(52**0.5)
print(Sharpe_hedged)

print(np.mean(hedged_ret)*52)
print(np.mean(hedged_excess_ret)*52)
print(np.std(hedged_ret)*(52**0.5))


# compute cumulative returns
#plt.plot(date,port_ret)

##portCumRetLong = np.cumprod(1+port['longonly'])
#portCumRetLongShort = np.cumprod(1+port['longshort'])

# plot graphs
'''
plt.figure(figsize=(10, 6))
plt.plot(date, portCumRetLong, label='Long-Only EW Portfolio')
plt.plot(date, portCumRetLongShort, label='Long-Short EW Portfolio')
plt.ylabel('Cumulative Return')
plt.title('Performance of Portfolios Constructed with Model on Weekly Frequency Data')
plt.legend(loc='lower right')

'''
"""
# view specific columns
cef(['ticker','volume'])
"""
