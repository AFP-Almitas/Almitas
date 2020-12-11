#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:01:43 2020
@author: kanp
"""
from cefpanelreg_fit import CEFpanelreg
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
- y: what you want to predict; default is 'cd', use 'cd5' for weekly cd
- var_pit: Point-in-time independent variables in [variable, lag]; unit in day
    e.g. ['volume',1] >> regress on lag1 of volume
- var_norm: Normalized independent variables in [variable, lag, length, func]; unit in day
    e.g. [cd,1,3,mean] >> regress on 3-day mean from lag1 of cd
    [cd,1,5,sum] >> equals to lag1 of weekly cd
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
y = ['cd5']
var_pit = []
var_norm = [['cd', 5, 5, 'sum'], ['pd', 1, 5, 'mean'], ['navchg', 1, 5, 'sum']]
fix = ['assetclasslevel1']
cluster = ['year', 'ticker']


"""
STEP3: Import data, train model, predict and compute SSE for each cross-validation sets
"""

# import data
data = pd.read_csv(filename)
data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])

# filter dates to include period in 1999 - 2015
regdata = data.loc[(data['date'] >= start_datetime)
                   & (data['date'] <= end_datetime)]
regdata = regdata.sort_values('date')
# asset = regdata[fix][:,1]).unique()

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
test_sets = []
starts = []
ends = []
test_starts = []
test_ends = []
all_SE = []
all_mse = []
all_train_asset = []
all_nobs = []
all_results = pd.DataFrame()

# store data for shrinkage test
all_validation = []
all_x = []
all_coef = []

for i in range(folds):
    reg_sets.append(regdata.loc[~regdata.date.isin(group_dates[i])])
    test_sets.append(regdata.loc[regdata.date.isin(group_dates[i])])
    starts.append(min(reg_sets[i].date).strftime('%Y-%m-%d'))
    ends.append(max(reg_sets[i].date).strftime('%Y-%m-%d'))
    test_starts.append(min(test_sets[i].date).strftime('%Y-%m-%d'))
    test_ends.append(max(test_sets[i].date).strftime('%Y-%m-%d'))

    # cef.to_csv(r'reg_file.csv')
    # test_sets[i].to_csv(r'test_file.csv')

    #filename_train = 'reg_file.csv'
    #filename_test = 'test_file.csv'

    cef = CEFpanelreg(reg_sets[i])
    cef.result(
        start_datetime=starts[i],
        end_datetime=ends[i],
        y=y,
        var_pit=var_pit,
        var_norm=var_norm,
        fix=fix,
        cluster=cluster
    )
    cef_test = CEFpanelreg(test_sets[i])
    cef_test.result(
        start_datetime=test_starts[i],
        end_datetime=test_ends[i],
        y=y,
        var_pit=var_pit,
        var_norm=var_norm,
        fix=fix,
        cluster=cluster
    )

    # backtest
    train_asset = pd.Series(
        cef.assetclass.iloc[:, 0]).sort_values().reset_index(drop=True)
    # drop first asset (the intercept)
    coef_asset = train_asset[1:].reset_index(drop=True)

    # filter dates
    start_datetime = test_starts[i]
    end_datetime = test_ends[i]
    validation = cef_test.data.loc[(cef_test.data['date'] >= start_datetime) & (
        cef_test.data['date'] <= end_datetime)]

    # filter columns
    validation = validation[y + ['year', 'ticker'] +
                            [col for col in validation.columns[cef.c:]] + fix + ['date', 'ret']]
    validation = validation.dropna()
    validation = validation.set_index(['ticker', 'year'])
    asset = pd.Series(validation[fix].iloc[:, 0].unique()
                      ).sort_values().reset_index(drop=True)

    # extract coefficients from fitted model
    fit = cef.result
    coef = fit._params

    # check if asset classes in validation set is also in the training data set
    check = np.zeros((len(asset)))
    for j in range(0, len(asset)):
        check[j] = any(train_asset.str.contains(asset[j]))

    # construct matrix of independent variables
    # start with array of ones for the intercept
    intercept = np.mat(np.repeat(1, len(validation)))

    # manually create columns of 1 for each asset class
    fix_asfactors = pd.DataFrame(np.zeros((len(validation), len(coef_asset))))
    assetclasscol = validation[fix].reset_index(drop=True)
    fix_asfactors = pd.concat([fix_asfactors, assetclasscol], axis=1)

    for j in range(0, len(coef_asset)):
        index = fix_asfactors[fix] == coef_asset[j]
        index = index.iloc[:, 0]
        fix_asfactors.loc[index, j] = 1

    # drop assetclasslevel3 column as it is no longer used
    fix_asfactors = fix_asfactors.drop(columns=fix)

    # select columns of independent variables
    indeptvar = validation.iloc[:, 1:-3]

    # construct the matrix of variables
    x = np.append(intercept.T, fix_asfactors, axis=1)
    x = np.append(x, indeptvar, axis=1)

    # predict cd based on coefficients from model fitted on training data set
    pred = pd.DataFrame(np.matmul(x, coef).T)

    # merge prediction with validation data set
    validation = validation.reset_index()
    validation = pd.concat([validation, pred], axis=1)
    validation = validation.rename({0: 'cdpred'}, axis='columns')

    # calculate sum of squared errors
    validation['diff'] = validation[y].iloc[:, 0] - validation['cdpred']
    validation['diff_sqr'] = validation['diff']**2

    SE = validation['diff_sqr'].sum()
    all_SE.append(SE)

    MSE = validation['diff_sqr'].mean()
    all_mse.append(MSE)

    # store results (SSE, r2, nobs and coef)
    res = pd.Series(
        np.append([SE, MSE, fit.rsquared, fit.nobs, test_starts[i], test_ends[i]], coef))
    rownames = pd.Series(np.append(
        ['SSE', 'MSE', 'r2', 'nobs', 'test set start date', 'test set ends date'], fit._var_names))
    if i == 0:
        all_results = pd.concat([rownames, res], axis=1)
        all_results.columns = ['row', 'val']
    else:
        results = pd.concat([rownames, res], axis=1)
        results.columns = ['row', 'val']

        all_results = all_results.merge(results, how='left', on='row')

    all_nobs.append(fit.nobs)

    # store data
    all_validation.append(validation)
    all_x.append(x)
    all_coef.append(coef)

    # store standard errors of the independent variables betas
    if i == 0:
        sigma = pd.DataFrame(fit.std_errors[-len(indeptvar.columns):])
    else:
        sigma = pd.concat(
            [sigma, fit.std_errors[-len(indeptvar.columns):]], axis=1)


all_results = all_results.set_index(['row'])
all_results.columns = list(range(1, folds+1))

# total SSE/nobs vs range of CD
total_sse = sum(all_SE)
total_nobs = sum(all_nobs)
sse_n = total_sse/total_nobs


print(all_SE)
print(sum(all_SE))
print(sum(all_mse))
print(np.std(all_mse))
print(all_results)

# shrinkage estimator for the coefficients on independent variables
# try use grand mean as shrinkage target
# compute omega squared as average of all std. errors from all folds
omega_sq = (sigma**2).mean(axis=1)

# compute grand mean of coefficients
for i in range(folds):
    if i == 0:
        beta = pd.DataFrame(all_coef[i][-len(indeptvar.columns):])
    else:
        beta = pd.concat(
            [beta, pd.Series(all_coef[i][-len(indeptvar.columns):])], axis=1)

beta['mean'] = beta.mean(axis=1)

# compute omega squared + delta squared as deviations of beta from grand mean
for i in range(folds):
    if i == 0:
        beta['total_deviation'] = (beta.iloc[:, i] - beta['mean'])**2
    else:
        beta['total_deviation'] = beta['total_deviation'] + \
            (beta.iloc[:, i] - beta['mean'])**2

beta['total_deviation'] = beta['total_deviation']/folds

# compute shrinkage intensity
shrinkage_intensity = 1 - \
    omega_sq.reset_index(drop=True) / beta['total_deviation']

# truncate at 0 for negative intensity
shrinkage_intensity[shrinkage_intensity < 0] = 0

# compute shrinkage estimator, new prediction, MSE, SSE, etc.
all_new_SE = []
all_new_mse = []

for i in range(folds):
    # shrinkage estimator
    beta['new'+str(i+1)] = beta.iloc[:, i]*shrinkage_intensity + \
        beta['mean']*(1-shrinkage_intensity)

    # replace coef
    new_coef = all_coef[i]
    new_coef[-len(indeptvar.columns):] = beta['new'+str(i+1)]

    # predict
    new_pred = pd.DataFrame(np.matmul(all_x[i], new_coef).T)

    # merge prediction with validation data set
    new_validation = all_validation[i]
    new_validation = pd.concat([new_validation, new_pred], axis=1)
    new_validation = new_validation.rename({0: 'new_cdpred'}, axis='columns')

    # calculate sum of squared errors
    new_validation['new_diff'] = new_validation[y].iloc[:, 0] - \
        new_validation['new_cdpred']
    new_validation['new_diff_sqr'] = new_validation['new_diff']**2

    SE = new_validation['new_diff_sqr'].sum()
    all_new_SE.append(SE)

    MSE = new_validation['new_diff_sqr'].mean()
    all_new_mse.append(MSE)


# try use 0 as shrinkage target
# compute omega squared as average of all std. errors from all folds
omega_sq = (sigma**2).mean(axis=1)

# compute omega squared + delta squared as deviations of beta from shrinkage target (0)
for i in range(folds):
    if i == 0:
        beta['total_deviation_2'] = (beta.iloc[:, i] - 0)**2
    else:
        beta['total_deviation_2'] = beta['total_deviation'] + \
            (beta.iloc[:, i] - 0)**2

beta['total_deviation_2'] = beta['total_deviation_2']/folds

# compute shrinkage intensity
shrinkage_intensity_2 = 1 - \
    omega_sq.reset_index(drop=True) / beta['total_deviation_2']

# truncate at 0 for negative intensity
# same shrinkage intensity so results will be the same..
shrinkage_intensity_2[shrinkage_intensity_2 < 0] = 0

# compute shrinkage estimator, new prediction, MSE, SSE, etc.
all_new_SE_2 = []
all_new_mse_2 = []

for i in range(folds):
    # shrinkage estimator
    beta['new_2'+str(i+1)] = beta.iloc[:, i] * \
        shrinkage_intensity_2 + 0*(1-shrinkage_intensity_2)

    # replace coef
    new_coef_2 = all_coef[i]
    new_coef_2[-len(indeptvar.columns):] = beta['new_2'+str(i+1)]

    # predict
    new_pred_2 = pd.DataFrame(np.matmul(all_x[i], new_coef_2).T)

    # merge prediction with validation data set
    new_validation = all_validation[i]
    new_validation = pd.concat([new_validation, new_pred_2], axis=1)
    new_validation = new_validation.rename({0: 'new_cdpred_2'}, axis='columns')

    # calculate sum of squared errors
    new_validation['new_diff_2'] = new_validation[y].iloc[:,
                                                          0] - new_validation['new_cdpred_2']
    new_validation['new_diff_sqr_2'] = new_validation['new_diff_2']**2

    SE = new_validation['new_diff_sqr_2'].sum()
    all_new_SE_2.append(SE)

    MSE = new_validation['new_diff_sqr_2'].mean()
    all_new_mse_2.append(MSE)

print(sum(all_SE))
print(sum(all_new_SE))
print(sum(all_new_SE_2))

print(np.std(all_mse))
print(np.std(all_new_mse))
print(np.std(all_new_mse_2))


# confusion matrix
# panelOLS.predict
== == == =
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:01:43 2020
@author: kanp
"""

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


data = pd.read_csv(filename, index_col=0)

data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])
alldates = data['date'].unique()

folds = 5

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
test_sets = []
starts = []
ends = []
test_starts = []
test_ends = []
all_SE = []

for i in range(5):
    reg_sets.append(data.loc[~data.date.isin(group_dates[i])])
    test_sets.append(data.loc[data.date.isin(group_dates[i])])
    starts.append(min(reg_sets[i].date).strftime('%Y-%m-%d'))
    ends.append(max(reg_sets[i].date).strftime('%Y-%m-%d'))
    test_starts.append(min(test_sets[i].date).strftime('%Y-%m-%d'))
    test_ends.append(max(test_sets[i].date).strftime('%Y-%m-%d'))

    reg_sets[i].to_csv(r'reg_file.csv')
    test_sets[i].to_csv(r'test_file.csv')

    filename_train = 'reg_file.csv'
    filename_test = 'test_file.csv'

    cef = CEFpanelreg(filename_train)
    cef.result(
        start_datetime=starts[i],
        end_datetime=ends[i],
        y=['cd'],
        var_pit=[['cd', 1], ['pd', 1], ['navchg', 1]],
        var_norm=[['cd', 1, 2, 'mean']],
        fix=['assetclasslevel3'],
        cluster=['year', 'ticker']
    )
    cef_test = CEFpanelreg(filename_test)
    cef_test.result(
        start_datetime=test_starts[i],
        end_datetime=test_ends[i],
        y=['cd'],
        var_pit=[['cd', 1], ['pd', 1], ['navchg', 1]],
        var_norm=[['cd', 1, 2, 'mean']],
        fix=['assetclasslevel3'],
        cluster=['year', 'ticker']
    )

    # backtest
    train_asset = pd.Series(
        cef.assetclass.iloc[:, 0]).sort_values().reset_index(drop=True)
    # drop first asset (the intercept)
    coef_asset = train_asset[1:].reset_index(drop=True)

    # filter dates
    start_datetime = test_starts[i]
    end_datetime = test_ends[i]
    validation = cef_test.data.loc[(cef_test.data['date'] >= start_datetime) & (
        cef_test.data['date'] <= end_datetime)]

    # filter columns
    y = ['cd']
    fix = ['assetclasslevel3']
    validation = validation[y + ['year', 'ticker'] +
                            [col for col in validation.columns[cef.c:]] + fix + ['date', 'ret']]
    validation = validation.dropna()
    validation = validation.set_index(['ticker', 'year'])
    asset = pd.Series(validation.assetclasslevel3.unique()
                      ).sort_values().reset_index(drop=True)

    # extract coefficients from fitted model
    fit = cef.result
    coef = fit._params

    # check if asset classes in validation set is also in the training data set
    check = np.zeros((len(asset)))
    for i in range(0, len(asset)):
        check[i] = any(train_asset.str.contains(asset[i]))

    # construct matrix of independent variables
    # start with array of ones for the intercept
    intercept = np.mat(np.repeat(1, len(validation)))

    # manually create columns of 1 for each asset class
    fix_asfactors = pd.DataFrame(np.zeros((len(validation), len(coef_asset))))
    assetclasscol = validation[fix].reset_index(drop=True)
    fix_asfactors = pd.concat([fix_asfactors, assetclasscol], axis=1)

    for i in range(0, len(coef_asset)):
        index = fix_asfactors[fix] == coef_asset[i]
        index = index.iloc[:, 0]
        fix_asfactors.loc[index, i] = 1

    # drop assetclasslevel3 column as it is no longer used
    fix_asfactors = fix_asfactors.drop(columns=fix)

    # select columns of independent variables
    indeptvar = validation.iloc[:, 1:-3]

    # construct the matrix of variables
    x = np.append(intercept.T, fix_asfactors, axis=1)
    x = np.append(x, indeptvar, axis=1)

    # predict cd based on coefficients from model fitted on training data set
    pred = pd.DataFrame(np.matmul(x, coef).T)

    # merge prediction with validation data set
    validation = validation.reset_index()
    validation = pd.concat([validation, pred], axis=1)
    validation = validation.rename({0: 'cdpred'}, axis='columns')

    validation['diff'] = validation['cd'] - validation['cdpred']
    validation['diff_sqr'] = validation['diff']**2

    SE = validation['diff_sqr'].sum()

    all_SE.append(SE)

print(all_SE)

'''
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

'''
"""
# view specific columns
cef(['ticker','volume'])
"""
== == == =
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:01:43 2020
@author: kanp
"""

"""
STEP1: Input file name
"""

#filename = 'mergedWeekly.csv'
filename = 'US_data.csv'

"""
STEP2: Input parameters for regression
- start datetime, end datetime in 'YYYY-MM-DD'
- folds: number of cross-validation sets
- y: what you want to predict; default is 'cd', use 'cd5' for weekly cd
- var_pit: Point-in-time independent variables in [variable, lag]; unit in day
    e.g. ['volume',1] >> regress on lag1 of volume
- var_norm: Normalized independent variables in [variable, lag, length, func]; unit in day
    e.g. [cd,1,3,mean] >> regress on 3-day mean from lag1 of cd
    [cd,1,5,sum] >> equals to lag1 of weekly cd
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
y = ['cd5']
var_pit = []
var_norm = [['cd', 5, 5, 'sum'], ['pd', 1, 5, 'mean'], ['navchg', 1, 5, 'sum']]
fix = ['assetclasslevel1']
cluster = ['year', 'ticker']


"""
STEP3: Import data, train model, predict and compute SSE for each cross-validation sets

"""

# import data
data = pd.read_csv(filename, index_col=0)
data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])

# filter dates to include period in 1999 - 2015
regdata = data.loc[(data['date'] >= start_datetime)
                   & (data['date'] <= end_datetime)]
regdata = regdata.sort_values('date')
# asset = regdata[fix][:,1]).unique()

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
test_sets = []
starts = []
ends = []
test_starts = []
test_ends = []
all_SE = []
all_nobs = []
all_results = pd.DataFrame()

for i in range(folds):
    reg_sets.append(regdata.loc[~regdata.date.isin(group_dates[i])])
    test_sets.append(regdata.loc[regdata.date.isin(group_dates[i])])
    starts.append(min(reg_sets[i].date).strftime('%Y-%m-%d'))
    ends.append(max(reg_sets[i].date).strftime('%Y-%m-%d'))
    test_starts.append(min(test_sets[i].date).strftime('%Y-%m-%d'))
    test_ends.append(max(test_sets[i].date).strftime('%Y-%m-%d'))

    # cef.to_csv(r'reg_file.csv')
    # test_sets[i].to_csv(r'test_file.csv')

    #filename_train = 'reg_file.csv'
    #filename_test = 'test_file.csv'

    cef = CEFpanelreg(reg_sets[i])
    cef.result(
        start_datetime=starts[i],
        end_datetime=ends[i],
        y=y,
        var_pit=var_pit,
        var_norm=var_norm,
        fix=fix,
        cluster=cluster
    )
    cef_test = CEFpanelreg(test_sets[i])
    cef_test.result(
        start_datetime=test_starts[i],
        end_datetime=test_ends[i],
        y=y,
        var_pit=var_pit,
        var_norm=var_norm,
        fix=fix,
        cluster=cluster
    )

    # backtest
    train_asset = pd.Series(
        cef.assetclass.iloc[:, 0]).sort_values().reset_index(drop=True)
    # drop first asset (the intercept)
    coef_asset = train_asset[1:].reset_index(drop=True)

    # filter dates
    start_datetime = test_starts[i]
    end_datetime = test_ends[i]
    validation = cef_test.data.loc[(cef_test.data['date'] >= start_datetime) & (
        cef_test.data['date'] <= end_datetime)]

    # filter columns
    validation = validation[y + ['year', 'ticker'] +
                            [col for col in validation.columns[cef.c:]] + fix + ['date', 'ret']]
    validation = validation.dropna()
    validation = validation.set_index(['ticker', 'year'])
    asset = pd.Series(validation[fix].iloc[:, 0].unique()
                      ).sort_values().reset_index(drop=True)

    # extract coefficients from fitted model
    fit = cef.result
    coef = fit._params

    # check if asset classes in validation set is also in the training data set
    check = np.zeros((len(asset)))
    for j in range(0, len(asset)):
        check[j] = any(train_asset.str.contains(asset[j]))

    # construct matrix of independent variables
    # start with array of ones for the intercept
    intercept = np.mat(np.repeat(1, len(validation)))

    # manually create columns of 1 for each asset class
    fix_asfactors = pd.DataFrame(np.zeros((len(validation), len(coef_asset))))
    assetclasscol = validation[fix].reset_index(drop=True)
    fix_asfactors = pd.concat([fix_asfactors, assetclasscol], axis=1)

    for j in range(0, len(coef_asset)):
        index = fix_asfactors[fix] == coef_asset[j]
        index = index.iloc[:, 0]
        fix_asfactors.loc[index, j] = 1

    # drop assetclasslevel3 column as it is no longer used
    fix_asfactors = fix_asfactors.drop(columns=fix)

    # select columns of independent variables
    indeptvar = validation.iloc[:, 1:-3]

    # construct the matrix of variables
    x = np.append(intercept.T, fix_asfactors, axis=1)
    x = np.append(x, indeptvar, axis=1)

    # predict cd based on coefficients from model fitted on training data set
    pred = pd.DataFrame(np.matmul(x, coef).T)

    # merge prediction with validation data set
    validation = validation.reset_index()
    validation = pd.concat([validation, pred], axis=1)
    validation = validation.rename({0: 'cdpred'}, axis='columns')

    # calculate sum of squared errors
    validation['diff'] = validation[y].iloc[:, 0] - validation['cdpred']
    validation['diff_sqr'] = validation['diff']**2

    SE = validation['diff_sqr'].sum()
    all_SE.append(SE)

    # keep track the coef's that deliver the lowest SE
    if len(all_SE) > 0:
        if SE <= min(all_SE):
            lowest_SE = SE
            best_coef = coef
            the_validation = validation

    # store results (SSE, r2, nobs and coef)
    res = pd.Series(
        np.append([SE, fit.rsquared, fit.nobs, test_starts[i], test_ends[i]], coef))
    rownames = pd.Series(np.append(
        ['SSE', 'r2', 'nobs', 'test set start date', 'test set ends date'], fit._var_names))
    if i == 0:
        all_results = pd.concat([rownames, res], axis=1)
        all_results.columns = ['row', 'val']
    else:
        results = pd.concat([rownames, res], axis=1)
        results.columns = ['row', 'val']

        all_results = all_results.merge(results, how='left', on='row')

    all_nobs.append(fit.nobs)


all_results = all_results.set_index(['row'])
all_results.columns = list(range(1, folds+1))

# total SSE/nobs vs range of CD
total_sse = sum(all_SE)
total_nobs = sum(all_nobs)
sse_n = total_sse/total_nobs

print(all_SE)
print(sum(all_SE))
print(total_nobs)
print(sse_n)
print(all_results)

# confusion matrix
cd_actl = the_validation[y].iloc[:, 0]  # actual
cd_pred = the_validation['cdpred']  # pred
the_validation['cdDirecPred'] = the_validation.groupby('ticker')[
    'cdpred'].pct_change()
the_validation['cdDirecActl'] = the_validation.groupby(
    'ticker').cd5.pct_change()
conf_mat = the_validation[['cdDirecPred', 'cdDirecActl']]
conf_mat.loc[conf_mat['cdDirecPred'] >= 0, 'pred'] = 1
conf_mat.loc[conf_mat['cdDirecPred'] < 0, 'pred'] = 0
conf_mat.loc[conf_mat['cdDirecActl'] >= 0, 'actl'] = 1
conf_mat.loc[conf_mat['cdDirecActl'] < 0, 'actl'] = 0

conf_mat.loc[(conf_mat['pred'] == 1) & (
    conf_mat['actl'] == 1), 'result'] = 'tp'
conf_mat.loc[(conf_mat['pred'] == 0) & (
    conf_mat['actl'] == 0), 'result'] = 'tn'
conf_mat.loc[(conf_mat['pred'] == 1) & (
    conf_mat['actl'] == 0), 'result'] = 'fp'
conf_mat.loc[(conf_mat['pred'] == 0) & (
    conf_mat['actl'] == 1), 'result'] = 'fn'

TP = sum(conf_mat['result'] == 'tp')
TN = sum(conf_mat['result'] == 'tn')
FP = sum(conf_mat['result'] == 'fp')
FN = sum(conf_mat['result'] == 'fn')

print([[TP, FP], [FN, TN]])

# plotting the residuals
