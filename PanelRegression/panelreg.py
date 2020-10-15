#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:01:43 2020

@author: kanp
"""

from cefpanelreg import CEFpanelreg

"""
STEP1: Input file name
"""

filename = 'mergedWeekly.csv'

"""
STEP2: Input parameters for regression
- start datetime, end datetime in 'YYYY-MM-DD'
- y: what you want to predict; default is 'cd'
- var_pit: Point-in-time independent variables in [variable, lag]; unit in day
    e.g. ['volume',1] >> regress on lag1 of volume
- var_norm: Normalized independent variables in [variable, lag, length, func]; unit in day
    e.g. [cd,1,3,mean] >> regress on 3-day mean from lag1 of cd
- fix: Fixed effects; choose one from ['assetclasslevel1','assetclasslevel2','assetclasslevel3']
- Cluster: Covariance clustering; choose from ['year','ticker']
"""

cef = CEFpanelreg(filename)
cef.result(
        start_datetime = '2013-01-01',
        end_datetime = '2016-12-31',
        y = ['cd'],
        var_pit = [['cd',1], ['pd',1], ['navchg',1]],
        var_norm = [['cd',1,2,'mean']],
        fix = ['assetclasslevel3'],
        cluster = ['year','ticker']
        )
cef.summary()


# backtest
train_asset = pd.Series(cef.assetclass.iloc[:,0]).sort_values().reset_index(drop=True)
# drop first asset (the intercept)
coef_asset = train_asset[1:].reset_index(drop=True)
    
# filter dates
start_datetime = '2017-01-01'
end_datetime = '2017-12-31'
validation = cef.data.loc[(cef.data['date']>=start_datetime) & (cef.data['date']<=end_datetime)]

# filter columns
y = ['cd']
fix = ['assetclasslevel3']
validation = validation[y + ['year','ticker'] + [col for col in validation.columns[cef.c:]] + fix + ['date', 'ret']]
validation = validation.dropna()
validation = validation.set_index(['ticker','year'])
asset = pd.Series(validation.assetclasslevel3.unique()).sort_values().reset_index(drop=True)

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

# form portfolios and compute weekly returns
date = validation['date'].unique()
port = pd.DataFrame(np.zeros((len(date), 2)), columns=['longonly', 'longshort'])
for t in range(0, len(date)) : 
    # sort CEFs into decile in each week, based on cdpred
    dt = validation.loc[validation['date']==date[t]]
    dt['decile'] = pd.qcut(validation['cdpred'], 10, labels=False)
    
    # form long only EW portfolio from the top decile
    port.loc[t, 'longonly'] = np.mean(dt.loc[dt.decile==9]['ret'])
    
    # form long-short EW portfolio from the top and bottom decile
    port.loc[t, 'longshort'] = np.mean(dt.loc[dt.decile==9]['ret']) - \
        np.mean(dt.loc[dt.decile==0]['ret'])

# compute cumulative returns
portCumRetLong = np.cumprod(1+port['longonly'])
portCumRetLongShort = np.cumprod(1+port['longshort'])

# plot graphs
plt.figure(figsize=(10, 6))
plt.plot(date, portCumRetLong, label='Long-Only EW Portfolio')
plt.plot(date, portCumRetLongShort, label='Long-Short EW Portfolio')
plt.ylabel('Cumulative Return')
plt.title('Performance of Portfolios Constructed with Model on Weekly Frequency Data')
plt.legend(loc='lower right')


"""

# view specific columns
cef(['ticker','volume'])

"""
