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
from cefpanelreg1 import CEFpanelreg
from datetime import datetime, date, timedelta

"""
STEP1: Input file name
"""

#filename = 'mergedWeekly.csv'
filename = 'US_data.csv'

"""
STEP2: Input parameters for regression
- start datetime, end datetime in 'YYYY-MM-DD'
- folds: number of cross-validation sets
- y: what you want to predict; default is 'cd'
- var_pit: Point-in-time independent variables in [variable, lag]; unit in day
    e.g. ['volume',1] >> regress on lag1 of volume
- var_norm: Normalized independent variables in [variable, lag, length, func]; unit in day
    e.g. [cd,1,3,mean] >> regress on 3-day mean from lag1 of cd
- fix: Fixed effects; choose one from ['assetclasslevel1','assetclasslevel2','assetclasslevel3']
- Cluster: Covariance clustering; choose from ['year','ticker']
"""

# input 
# start and end dates of training/validation set
start_datetime = '1999-01-01'
end_datetime = '2015-12-31'
# number of cross-validation sets
folds = 5
# parameters for panel reg
var_pit = [['cd',1], ['pd',1], ['navchg',1]]
var_norm = [['cd',1,2,'mean']]
fix = ['assetclasslevel3']
cluster = ['year','ticker']


"""
STEP3: Import data, train model, predict and compute SSE for each cross-validation sets

"""

# import data
data = pd.read_csv(filename, index_col = 0)
data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])

# filter dates to include period in 1999 - 2015
regdata = data.loc[(data['date']>=start_datetime) & (data['date']<=end_datetime)]
regdata = regdata.sort_values('date')
#asset = regdata[fix][:,1]).unique()

alldates = regdata['date'].unique()
group_len = math.floor(len(alldates)/folds)
group_dates = []

for i in range(folds):
    if i == 0:
        group_dates.append(alldates[0:group_len])
    elif i == folds - 1:
        group_dates.append(alldates[(group_len*(i)):len(alldates)])
    else:
        group_dates.append(alldates[(group_len*(i)):(group_len*(i+1))])

reg_sets = []
test_sets= []
starts = []
ends = []
test_starts = []
test_ends = []
all_SE = []
all_results = pd.DataFrame()

for i in range(folds):
    reg_sets.append(regdata.loc[~regdata.date.isin(group_dates[i])])
    test_sets.append(regdata.loc[regdata.date.isin(group_dates[i])])
    starts.append(min(reg_sets[i].date).strftime('%Y-%m-%d'))
    ends.append(max(reg_sets[i].date).strftime('%Y-%m-%d'))
    test_starts.append(min(test_sets[i].date).strftime('%Y-%m-%d'))
    test_ends.append(max(test_sets[i].date).strftime('%Y-%m-%d'))

    #cef.to_csv(r'reg_file.csv')
    #test_sets[i].to_csv(r'test_file.csv')

    #filename_train = 'reg_file.csv'
    #filename_test = 'test_file.csv'

    cef = CEFpanelreg(reg_sets[i])
    cef.result(
            start_datetime = starts[i],
            end_datetime = ends[i],
            y = ['cd'],
            var_pit = var_pit,
            var_norm = var_norm,
            fix = fix,
            cluster = cluster
            )
    cef_test = CEFpanelreg(test_sets[i])
    cef_test.result(
            start_datetime = test_starts[i],
            end_datetime = test_ends[i],
            y = ['cd'],
            var_pit = var_pit,
            var_norm = var_norm,
            fix = fix,
            cluster = cluster
            )
    
    # backtest
    train_asset = pd.Series(cef.assetclass.iloc[:,0]).sort_values().reset_index(drop=True)
    # drop first asset (the intercept)
    coef_asset = train_asset[1:].reset_index(drop=True)

    # filter dates
    start_datetime = test_starts[i]
    end_datetime = test_ends[i]
    validation = cef_test.data.loc[(cef_test.data['date']>=start_datetime) & (cef_test.data['date']<=end_datetime)]

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
    for j in range(0, len(asset)) :
        check[j] = any(train_asset.str.contains(asset[j]))

    # construct matrix of independent variables
    # start with array of ones for the intercept
    intercept = np.mat(np.repeat(1,len(validation)))

    # manually create columns of 1 for each asset class
    fix_asfactors = pd.DataFrame(np.zeros((len(validation), len(coef_asset))))
    assetclasscol = validation[fix].reset_index(drop=True)
    fix_asfactors = pd.concat([fix_asfactors, assetclasscol], axis=1)

    for j in range(0, len(coef_asset)) : 
        index = fix_asfactors[fix]==coef_asset[j]
        index = index.iloc[:,0] 
        fix_asfactors.loc[index,j] = 1

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
    
    # calculate sum of squared errors
    validation['diff'] = validation['cd'] - validation['cdpred']
    validation['diff_sqr'] = validation['diff']**2

    SE = validation['diff_sqr'].sum()
    all_SE.append(SE)
    
    # store results (SSE, r2, nobs and coef)
    res = pd.Series(np.append([SE, fit.rsquared, fit.nobs, starts[i], ends[i]], coef))
    rownames = pd.Series(np.append(['SSE', 'r2', 'nobs'], fit._var_names))
    if i==0 :        
        all_results = pd.concat([rownames, res], axis=1)
        all_results.columns = ['row', 'val']
    else :
        results = pd.concat([rownames, res], axis=1)
        results.columns = ['row', 'val']
        
        all_results = all_results.merge(results, how='left', on='row')
    

all_results = all_results.set_index(['row'])
all_results.columns= list(range(1,folds+1))

print(all_SE)
print(sum(all_SE))
print(all_results)
