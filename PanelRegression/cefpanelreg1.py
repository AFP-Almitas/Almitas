#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:48:04 2020

@author: kanp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from datetime import datetime, date, timedelta
from linearmodels.panel import PanelOLS
from linearmodels.panel import compare
import sys
sys.tracebacklimit = 0
np.warnings.filterwarnings('ignore')

class CEFpanelreg:
    
    def __init__(self, data):
        #self.filename = str(file)
        print('Importing data...')
        #data = pd.read_csv(self.filename, index_col = 0)
        data.columns = [x.lower() for x in data.columns]
        data['date'] = pd.to_datetime(data['date'])        
        print('Import success')
        self.data = data
        self.sumstat = {}
        
    def __call__(self, collist):
        self.__Checkcol(collist, self.data)
    
    def result(self,
               start_datetime = datetime(2020, 1, 1),
               end_datetime = datetime(2000, 1, 1),
               y = ['cd'],
               var_pit = [],
               var_norm = [],
               fix = [],                
               cluster = []
               ):

        # raise error when start-end date format
        self.__Checkdate(start_datetime, end_datetime)
        
        # date variables
        self.data['year'] = pd.DatetimeIndex(self.data['date']).year
        self.data['month'] = pd.DatetimeIndex(self.data['date']).month
        self.data['inceptiondate'] = pd.to_datetime(self.data['inceptiondate'])
        self.data['terminationdate'] = pd.to_datetime(self.data['terminationdate'])
        self.data = self.data.drop_duplicates(subset=['ticker','date'],keep='last')
        
        # check valid lags, drop later before regression
        self.__CheckValidLag()
        
        # compute return
        self.data['lpriceclose'] = self.data[['ticker','priceclose']].groupby('ticker').shift(1)  
        self.data['ret'] = self.data['priceclose']/self.data['lpriceclose'] - 1
        
        # change in discount
        #self.data['lpd'] = self.data[['ticker','pd']].groupby('ticker').shift(5)  #1
        self.data['lpd'] = self.data[['ticker','pd']].groupby('ticker').shift(1)  #2
        self.data['cd'] = self.data['pd'] - self.data['lpd']
        
        # age
        self.data['age'] = (self.data['date']-self.data['inceptiondate']).dt.days
        self.data['age'] = np.log(self.data['age'])

        # time to maturity
        #self.data.loc[self.data['terminationdate'].isna() & (self.data['marketstatus']=='ACTV')]['terminationdate'] = '2019-12-31'
        self.data['tomaturity'] = (self.data['terminationdate']-self.data['date']).dt.days
        self.data['tomaturity'] = np.log(self.data['tomaturity'])
    
        # nav change
        self.data['lnav'] = self.data[['ticker','nav']].groupby('ticker').shift(1)      #1a/2
        #self.data['lnav'] = self.data[['ticker','nav']].groupby('ticker').shift(5)      #1b
        self.data['navchg'] = self.data['nav'] - self.data['lnav']
       
        # select only rows that has valid weekly data
        #self.data = self.data.loc[self.data.validweekly==True]     #2

        # column reference
        self.c = len(self.data.columns)
        
        # Point-in-time variables
        for var in var_pit:
            variable,lag = var[0],var[1]
            self.data[variable+'_'+str(lag)] = self.data.groupby(['ticker'])[variable].shift(lag)
        
        # Normalized variables
        def decor(func):
            def wrapper(x,y,z,var):
                print("Regressing on "+str(var)+" using "+str(z)+"-day "+func.__name__+" from lag "+str(y)+"...")
                return func(x,y,z,var)
            return wrapper
        @decor
        def mean(df,lag,length,var):
            df[variable+'_'+str(lag)+'_'+f+str(length)] = df.groupby('ticker')[variable].shift(lag).rolling(length).mean()
        @decor
        def std(df,lag,length,var):
            df[variable+'_'+str(lag)+'_'+f+str(length)] = df.groupby('ticker')[variable].shift(lag).rolling(length).std()
            
        func_dict = {'mean':mean,
                     'std':std}
        
        for var in var_norm:
            variable,lag,length,f = var[0],var[1],var[2],var[3]
            func_dict[f](self.data,lag,length,variable)
        
        # fit regression and return results
        self.result = self.__fitreg(self.data, start_datetime, end_datetime, y, var_pit, var_norm, fix, cluster, self.c)
        
        # extract .nobs, .rsquared, .params, .tstats
        self.sumstat['R2'] = round(self.result.rsquared,4)
        self.sumstat['N'] = self.result.nobs
        self.sumstat['Coefficient'] = round(self.result.params,4)
        self.sumstat['t-stat'] = round(self.result.tstats,4)
        
        return self.result

    def summary(self):
        
        print(pd.concat([self.sumstat['Coefficient'],self.sumstat['t-stat']], axis = 1))
        print("R2 = {}".format(self.sumstat['R2']))
        print("N = {}".format(self.sumstat['N']))

        #temp = df1.sort_values('date').groupby(['ticker','year','month']).tail(1) #use last observation of the month
        #temp = temp[['ticker','year','month','volume']]
        #temp = temp.rename(columns = {'volume':'volume'+'-'+str(lag)+str(unit)})
        #df1 = pd.merge(df, temp, how='left', on=['ticker','year','month'])
        
        
    def __Checkcol(self, collist, df):
        count = 0
        for column in collist:
            if column not in df.columns:
                pass
            else:
                count += 1   
        if count != len(collist):
            raise TypeError('Some columns are not available!')
        else:
            print(df[collist])
            
    def __Checkdate(self, start_datetime, end_datetime):
        start_datetime = pd.to_datetime(start_datetime)
        end_datetime = pd.to_datetime(end_datetime)
        if start_datetime > datetime.now():
            raise ValueError(
                "Input start_datetime > current date; current date is {}".format(
                    datetime.now().strftime("%Y-%m-%d")
                )
            )
        if end_datetime > self.data['date'].max():
            raise ValueError(
                "Input end_datetime > latest available date; latest available date is {}".format(
                    self.data['date'].max()
                )
            )
        if start_datetime < self.data['date'].min():
            raise ValueError(
                "Input start_datetime < earliest available date; earliest available date is {}".format(
                    self.data['date'].min()
                )
            )
        if start_datetime > end_datetime:
            raise ValueError(
                "Input start_datetime > end_datetime; choose dates between {} and {}".format(
                    end_datetime.strftime("%Y-%m-%d"),
                    start_datetime.strftime("%Y-%m-%d"),
                )
            )

    def __CheckValidLag(self, lim=8):
        self.data['dif'] = self.data.groupby(['ticker'])['date'].diff()
        self.data['dif'] = self.data['dif'].fillna(pd.Timedelta(seconds=0)).dt.days.astype(int)
        self.data['valid'] = self.data['dif']<lim
        #self.data['valid'] = [d<limit for d in self.data['dif']]
        self.data.drop(['dif'], axis=1, inplace=True)

    def __fitreg(self,
                 dt,
                 start_datetime,
                 end_datetime,
                 y,
                 var_pit,
                 var_norm,
                 fix,
                 cluster,
                 c
                 ):

        # filter dates
        dt = dt.loc[(dt['date']>=start_datetime) & (dt['date']<=end_datetime)]

        # filter columns
        dt = dt[y + ['year','ticker'] + [col for col in dt.columns[c:]] + fix]
        
        # choose x
        x = '+'.join(dt.columns[3:])
        
        #print("Start filling NAs...")
        #dt = dt.fillna(dt.groupby('ticker').transform('mean'))
        #dt = dt.fillna(dt.transform('mean'))
        dt = dt.dropna()
        
        #print("Filling NAs done.")
        dt = dt.set_index(['ticker','year'])
        
        self.assetclass = dt[fix].drop_duplicates().reset_index(drop=True)

        print(dt.info())
          
        if len(fix) == 0 and len(cluster) == 0:
            mod = PanelOLS.from_formula(y[0] + '~1+' + x, data = dt)
            fit1 = mod.fit(cov_type = 'heteroskedastic', cluster_time = False, cluster_entity = False)
            print(fit1)
            return fit1
        
        if len(fix) == 1:
            mod = PanelOLS.from_formula(y[0] + '~1+' + x + '+' + fix[0], data = dt)
            if len(cluster) == 0:
                fit1 = mod.fit(cov_type = 'clustered', cluster_time = False, cluster_entity = False)
                print(fit1)
                return fit1
            elif cluster == ['year']:
                fit1 = mod.fit(cov_type = 'clustered', cluster_time = True, cluster_entity = False)
                print(fit1)
                return fit1
            elif cluster == ['ticker']:
                fit1 = mod.fit(cov_type = 'clustered', cluster_time = False, cluster_entity = True)
                print(fit1)
                return fit1
            elif cluster == ['year','ticker'] or cluster == ['ticker','year']:
                fit1 = mod.fit(cov_type = 'clustered', cluster_time = True, cluster_entity = True)
                print(fit1)
                return fit1
            else:
                raise KeyError("Please choose either year or ticker, or both.")
            
        if len(fix) > 1:
            raise KeyError("You have {} fixed effects! Please pick one.".format(len(fix)))

    