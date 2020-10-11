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
from sklearn.preprocessing import LabelBinarizer

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

fit = cef.result
coef = fit._params

indeptvar = validation.iloc[:,1:-3]
asset = validation.assetclasslevel3.unique()
fix_asfactors = LabelBinarizer().fit_transform(validation.assetclasslevel3)

intercept = np.mat(np.repeat(1,len(validation)))
x = np.append(intercept.T,fix_asfactors, axis=1)
x = np.append(x, indeptvar, axis=1)

pred = pd.DataFrame(np.matmul(x, coef).T)

validation = validation.reset_index()
validation = pd.concat([validation, pred], axis=1)
validation= validation.rename({0: 'cdpred'}, axis='columns')

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
    
portCumRetLong = np.cumprod(1+port['longonly'])
portCumRetLongShort = np.cumprod(1+port['longshort'])

# plot graphs
plt.figure(figsize=(10, 6))
plt.plot(date, portCumRetLong, label='EW long only top-decile portfolio')
plt.plot(date, portCumRetLongShort, label='EW long-short (top and bottom decile) portfolio')
plt.ylabel('Cumulative Return')
plt.legend(loc='lower right')

#table = cef.data
#test: 2013-2016
#try premium/discount group seperately

"""

# view specific columns
cef(['ticker','volume'])

"""
