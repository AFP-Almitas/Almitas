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
filename = 'US_data_2.csv'

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

# import VIX data
vix = pd.read_csv("vix.csv")
vix['Date'] = pd.to_datetime(vix['Date'])
vix.columns = [x.lower() for x in vix.columns]

# merge vix data
data = data.merge(vix[['date', 'vix']], how='left', on='date')

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
all_conf_mat = []   

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
    all_train_asset.append(train_asset)
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
    
    # construct confusion matrix
    conf_mat = validation[y+['cdpred']]
    
    conf_mat.loc[(conf_mat.iloc[:,0] >= 0) & (conf_mat.iloc[:,1] >= 0), 'result'] = 'tp'
    conf_mat.loc[(conf_mat.iloc[:,0] < 0) & (conf_mat.iloc[:,1] < 0), 'result'] = 'tn'
    conf_mat.loc[(conf_mat.iloc[:,0] >= 0) & (conf_mat.iloc[:,1] < 0), 'result'] = 'fn'
    conf_mat.loc[(conf_mat.iloc[:,0] < 0) & (conf_mat.iloc[:,1] >= 0), 'result'] = 'fp'

    TP = sum(conf_mat['result'] == 'tp')
    TN = sum(conf_mat['result'] == 'tn')
    FP = sum(conf_mat['result'] == 'fp')
    FN = sum(conf_mat['result'] == 'fn')
    
    print([TP, FP]) 
    print([FN, TN])
    
    all_conf_mat.append([TP, TN, FP, FN])
    
    TPR = TP/(TP+FN)
    Precision = TP/(TP+FP)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)

    # store results (SSE,MSE, r2, nobs, date of test set and coef)
    res = pd.Series(np.append([SE, MSE, fit.rsquared, fit.nobs, test_starts[i], test_ends[i],
                               TPR, Precision, Accuracy], coef))
    rownames = pd.Series(np.append(['SSE', 'MSE', 'r2', 'nobs', 'test set start date', 'test set end date',
                                    'TPR', 'Precision', 'Accuracy'], fit._var_names))
    if i == 0:
        all_results = pd.concat([rownames, res], axis=1)
        all_results.columns = ['row', 'val']
    else:
        results = pd.concat([rownames, res], axis=1)
        results.columns = ['row', 'val']

        all_results = all_results.merge(results, how='left', on='row')

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

all_TP = [res[0] for res in all_conf_mat]
all_TN = [res[1] for res in all_conf_mat]
all_FP = [res[2] for res in all_conf_mat]
all_FN = [res[3] for res in all_conf_mat]

all_TPR = sum(all_TP)/sum(all_TP + all_FN)
all_precision = sum(all_TP)/sum(all_TP + all_FP)
all_accuracy = sum(all_TP + all_TN)/sum(all_TP + all_FP + all_TN + all_FN)

print(all_TPR)
print(all_precision)
print(all_accuracy)


# shrinkage estimator for the coefficients on independent variables
# try use grand mean as shrinkage target
# compute omega squared as average of all std. errors from all folds
#omega_sq = (sigma**2).mean(axis=1)
omega_sq = sigma**2

# compute grand mean of coefficients
for i in range(folds) :
    if i==0 :
        beta = pd.DataFrame(all_coef[i][-len(indeptvar.columns):])
    else :
        beta = pd.concat([beta, pd.Series(all_coef[i][-len(indeptvar.columns):])], axis=1)

beta['mean'] = beta.mean(axis=1)

# compute omega squared + delta squared as deviations of beta from grand mean
for i in range(folds) :
    if i==0 :
        beta['total_deviation'] = (beta.iloc[:,i] - beta['mean'])**2
    else : 
        beta['total_deviation'] = beta['total_deviation'] + (beta.iloc[:,i] - beta['mean'])**2

beta['total_deviation'] = beta['total_deviation']/folds

# compute shrinkage intensity
#shrinkage_intensity = 1 - omega_sq.reset_index(drop=True) / beta['total_deviation']
shrinkage_intensity = omega_sq.apply(lambda x: 1-x/list(beta['total_deviation']), axis=0)
shrinkage_intensity = shrinkage_intensity.reset_index(drop=True)

# truncate at 0 for negative intensity
shrinkage_intensity[shrinkage_intensity<0] = 0

# compute shrinkage estimator, new prediction, MSE, SSE, etc.
all_new_SE = []
all_new_mse = []
all_new_TPR = []
all_new_precision = []
all_new_accuracy = []
all_new_conf_mat = []

for i in range(folds) :
    # shrinkage estimator
    beta['new'+str(i+1)] = beta.iloc[:,i]*shrinkage_intensity.iloc[:,i] + beta['mean']*(1-shrinkage_intensity.iloc[:,i])
    
    # replace coef
    new_coef = all_coef[i]
    new_coef[-len(indeptvar.columns):] = beta['new'+str(i+1)]
    
    # predict
    new_pred = pd.DataFrame(np.matmul(all_x[i], new_coef).T)

    # merge prediction with validation data set
    new_validation = all_validation[i]
    new_validation = pd.concat([new_validation, new_pred], axis=1)
    new_validation= new_validation.rename({0: 'new_cdpred'}, axis='columns')
    
    # calculate sum of squared errors
    new_validation['new_diff'] = new_validation[y].iloc[:,0] - new_validation['new_cdpred']
    new_validation['new_diff_sqr'] = new_validation['new_diff']**2

    SE = new_validation['new_diff_sqr'].sum()
    all_new_SE.append(SE)

    MSE = new_validation['new_diff_sqr'].mean()
    all_new_mse.append(MSE)
    
    # construct confusion matrix
    conf_mat = new_validation[y+['cdpred']]
    
    conf_mat.loc[(conf_mat.iloc[:,0] >= 0) & (conf_mat.iloc[:,1] >= 0), 'result'] = 'tp'
    conf_mat.loc[(conf_mat.iloc[:,0] < 0) & (conf_mat.iloc[:,1] < 0), 'result'] = 'tn'
    conf_mat.loc[(conf_mat.iloc[:,0] >= 0) & (conf_mat.iloc[:,1] < 0), 'result'] = 'fn'
    conf_mat.loc[(conf_mat.iloc[:,0] < 0) & (conf_mat.iloc[:,1] >= 0), 'result'] = 'fp'

    TP = sum(conf_mat['result'] == 'tp')
    TN = sum(conf_mat['result'] == 'tn')
    FP = sum(conf_mat['result'] == 'fp')
    FN = sum(conf_mat['result'] == 'fn')
    
    print([TP, FP]) 
    print([FN, TN])
    
    all_new_conf_mat.append([TP, TN, FP, FN])
    
    TPR = TP/(TP+FN)
    Precision = TP/(TP+FP)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    all_new_TPR.append(TPR)
    all_new_precision.append(Precision)
    all_new_accuracy.append(Accuracy)
    

all_TP = [res[0] for res in all_new_conf_mat]
all_TN = [res[1] for res in all_new_conf_mat]
all_FP = [res[2] for res in all_new_conf_mat]
all_FN = [res[3] for res in all_new_conf_mat]

all_TPR = sum(all_TP)/sum(all_TP + all_FN)
all_precision = sum(all_TP)/sum(all_TP + all_FP)
all_accuracy = sum(all_TP + all_TN)/sum(all_TP + all_FP + all_TN + all_FN)

print(all_TPR)
print(all_precision)
print(all_accuracy)



# try use 0 as shrinkage target
# compute omega squared as average of all std. errors from all folds
omega_sq = sigma**2

# compute omega squared + delta squared as deviations of beta from shrinkage target (0)
for i in range(folds) :
    if i==0 :
        beta['total_deviation_2'] = (beta.iloc[:,i] - 0)**2
    else : 
        beta['total_deviation_2'] = beta['total_deviation'] + (beta.iloc[:,i] - 0)**2

beta['total_deviation_2'] = beta['total_deviation_2']/folds

# compute shrinkage intensity
shrinkage_intensity_2 = omega_sq.apply(lambda x: 1-x/list(beta['total_deviation_2']), axis=0)
shrinkage_intensity_2 = shrinkage_intensity_2.reset_index(drop=True)

# truncate at 0 for negative intensity
shrinkage_intensity_2[shrinkage_intensity_2<0] = 0 

# compute shrinkage estimator, new prediction, MSE, SSE, etc.
all_new_SE_2 = []
all_new_mse_2 = []
all_new_TPR_2 = []
all_new_precision_2 = []
all_new_accuracy_2 = []
all_new_conf_mat_2 = []

for i in range(folds) :
    # shrinkage estimator
    beta['new_2'+str(i+1)] = beta.iloc[:,i]*shrinkage_intensity_2.iloc[:,i] + 0*(1-shrinkage_intensity_2.iloc[:,i])
    
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
    new_validation['new_diff_2'] = new_validation[y].iloc[:,0] - new_validation['new_cdpred_2']
    new_validation['new_diff_sqr_2'] = new_validation['new_diff_2']**2

    SE = new_validation['new_diff_sqr_2'].sum()
    all_new_SE_2.append(SE)

    MSE = new_validation['new_diff_sqr_2'].mean()
    all_new_mse_2.append(MSE)
    
    # construct confusion matrix
    conf_mat = new_validation[y+['cdpred']]
    
    conf_mat.loc[(conf_mat.iloc[:,0] >= 0) & (conf_mat.iloc[:,1] >= 0), 'result'] = 'tp'
    conf_mat.loc[(conf_mat.iloc[:,0] < 0) & (conf_mat.iloc[:,1] < 0), 'result'] = 'tn'
    conf_mat.loc[(conf_mat.iloc[:,0] >= 0) & (conf_mat.iloc[:,1] < 0), 'result'] = 'fn'
    conf_mat.loc[(conf_mat.iloc[:,0] < 0) & (conf_mat.iloc[:,1] >= 0), 'result'] = 'fp'

    TP = sum(conf_mat['result'] == 'tp')
    TN = sum(conf_mat['result'] == 'tn')
    FP = sum(conf_mat['result'] == 'fp')
    FN = sum(conf_mat['result'] == 'fn')
    
    print([TP, FP]) 
    print([FN, TN])
    
    all_new_conf_mat_2.append([TP, TN, FP, FN])
    
    TPR = TP/(TP+FN)
    Precision = TP/(TP+FP)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    all_new_TPR_2.append(TPR)
    all_new_precision_2.append(Precision)
    all_new_accuracy_2.append(Accuracy)
    

print(sum(all_SE))
print(sum(all_new_SE))
print(sum(all_new_SE_2))

print(np.std(all_mse))
print(np.std(all_new_mse))
print(np.std(all_new_mse_2))

all_TP = [res[0] for res in all_new_conf_mat_2]
all_TN = [res[1] for res in all_new_conf_mat_2]
all_FP = [res[2] for res in all_new_conf_mat_2]
all_FN = [res[3] for res in all_new_conf_mat_2]

all_TPR = sum(all_TP)/sum(all_TP + all_FN)
all_precision = sum(all_TP)/sum(all_TP + all_FP)
all_accuracy = sum(all_TP + all_TN)/sum(all_TP + all_FP + all_TN + all_FN)

print(all_TPR)
print(all_precision)
print(all_accuracy)




# plotting the residuals
# how to plot them?
the_validation['res'] = the_validation[y] - the_validation['cdpred']
