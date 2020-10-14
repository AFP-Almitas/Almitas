#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:55:49 2020

@author: kanp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, date, timedelta
from cefpanelreg import CEFpanelreg

class CEFbacktest(CEFpanelreg):
            
    def backtest(self,
                 start_datetime = datetime(2010, 1, 1),
                 end_datetime = datetime(2020, 1, 1),
                 assetclass1 = 'Equity',
                 alpha = {'pit':[],
                          'norm':[['cd',5,5,'mean'],['volume',5,5,'mean']]},
                 weight = {'pit':[],
                          'norm':[0.5,0.5]},
                 method = 'scoring',
                 freq = ['weekly'], #['weekly'] or ['monthly',0 or 1]
                 compare = False
                 ):
        
        # raise error when start-end date format
        self.__Checkdate(start_datetime, end_datetime)
        
        # date variables
        self.data['year'] = pd.DatetimeIndex(self.data['date']).year
        self.data['month'] = pd.DatetimeIndex(self.data['date']).month
        self.data['week'] = pd.DatetimeIndex(self.data['date']).week
        self.data['inceptiondate'] = pd.to_datetime(self.data['inceptiondate'])
        self.data = self.data.drop_duplicates(subset=['ticker','date'],keep='last')
        
        # change in discount (try 5 days)
        self.data['lpd'] = self.data[['ticker','pd']].groupby('ticker').shift(5)
        self.data['cd'] = self.data['pd'] - self.data['lpd']
        
        # column reference
        c = len(self.data.columns)
        
        # Point-in-time variables
        for var in alpha['pit']:
            variable,lag = var[0],var[1]
            self.data[variable+'_'+str(lag)] = self.data.groupby(['ticker'])[variable].shift(lag)
        
        # Normalized variables
        def decor(func):
            def wrapper(x,y,z,var):
                print("Creating alpha "+str(var)+" using "+str(z)+"-day "+func.__name__+" from lag "+str(y)+"...")
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
        
        for var in alpha['norm']:
            variable,lag,length,f = var[0],var[1],var[2],var[3]
            func_dict[f](self.data,lag,length,variable)
            
        # filter dates
        df = self.data.loc[(self.data['date']>=start_datetime) & (self.data['date']<=end_datetime)]
        
        # filter asset class
        df = df.loc[df['assetclasslevel1']==assetclass1]
        
        # filter columns
        df = df[['year','month','week','date','ticker','priceclose','cd'] + [col for col in df.columns[c:]]]
        
        # filter weekly OR monthly
        if freq[0]=='weekly':
            #df['weekday'] = pd.to_datetime(df.date).dt.dayofweek
            #df = df.loc[df['weekday']==freq[1]]

            df = df.sort_values(by=['ticker','date']).groupby(['ticker','year','month','week']).first()
            df = df.reset_index()
            
            # weekly return            
            df['nextpriceclose'] = df.groupby('ticker')['priceclose'].shift(-1)
            df['ret'] = df['nextpriceclose']/df['priceclose'] - 1
            df = df.drop(['nextpriceclose'],1)
            cols = list(df.columns)
            cols = cols[:5]+[cols[-1]]+cols[5:-1]
            df = df[cols]
            
            # check lag if it's one week interval
            df['dif'] = df.sort_values(['ticker','date']).groupby(['ticker','year'])['week'].diff().shift(-1)
            #df['dif'] = df['dif'].fillna(pd.Timedelta(seconds=0)).dt.days.astype(int)
            df['valid'] = df['dif']==1
            df.drop(['dif'], axis=1, inplace=True)
            df.drop(['valid'], axis=1, inplace=True)
        
        elif freq[0]=='monthly':
            df.drop(['week'], 1, inplace=True)
            if freq[1]==1:
                df = df.sort_values(by=['ticker','date']).groupby(['ticker','year','month']).tail(1)
            elif freq[1]==0:
                df = df.sort_values(by=['ticker','date']).groupby(['ticker','year','month']).head(1)
                
            # monthly return
            df['nextpriceclose'] = df.groupby('ticker')['priceclose'].shift(-1)
            df['ret'] = df['nextpriceclose']/df['priceclose'] - 1
            df = df.drop(['nextpriceclose'],1)
            cols = list(df.columns)
            cols = cols[:5]+[cols[-1]]+cols[5:-1]
            df = df[cols]
            
            # check lag if it's one month interval
            df['dif'] = df.sort_values(['ticker','date']).groupby(['ticker','year'])['month'].diff().shift(-1)
            df['valid'] = df['dif']==1
            df.drop(['dif'], axis=1, inplace=True)
            df.drop(['valid'], axis=1, inplace=True)
        
        # backtest
        if method[0]=='scoring':
            self.result = self.backtest_scoring(df, method[1], method[2], method[3], freq[0])
            
        elif method[0]=='model':
            self.result = self.backtest_model(df)
            
        elif method[0]=='mve':
            self.result = self.backtest_mve(df)
        
        return self.result
        
    def backtest_scoring(self, df, factorweight, portweight, nquintile, freq):

        w = factorweight['pit'] + factorweight['norm']
        k = len(w)
        
        sig = df.columns[-k:]
        df = df.sort_values('date')
        
        for s in sig:
            if freq=='weekly':
                df[s+'_standard'] = df.groupby(['year','month','week'])[s].transform(lambda x: (x-x.mean())/x.std())
            elif freq=='monthly':
                df[s+'_standard'] = df.groupby(['year','month'])[s].transform(lambda x: (x-x.mean())/x.std())
            
            #df[s+'_rank'] = df.groupby('date')[s].rank('dense',ascending=False)
              
        alpha_standard = df[df.columns[-k:]]
        df['score'] = alpha_standard.dot(w)

        df = df.dropna()
        if freq=='weekly':
            df['decile'] = df.groupby(['year','month','week'])['score'].apply(pd.cut, bins=nquintile, labels=False)
        elif freq=='monthly':
            df['decile'] = df.groupby(['year','month'])['score'].apply(pd.cut, bins=nquintile, labels=False)

        df = df.loc[(df['decile']==0) | (df['decile']==nquintile-1)]

        if portweight=='ew':
            if freq=='weekly':
                df['N'] = df.groupby(['year','month','week','decile'])['decile'].transform(lambda x: x.count())
            elif freq=='monthly':
                df['N'] = df.groupby(['year','month','decile'])['decile'].transform(lambda x: x.count())
            df['weight'] = 1/df['N']
            df.drop(['N'], 1, inplace=True)
        elif portweight=='vw':
            if freq=='weekly':
                df['weight'] = df.groupby(['year','month','week','decile'])['priceclose'].transform(lambda x: x/x.sum())
            elif freq=='monthly':
                df['weight'] = df.groupby(['year','month','decile'])['priceclose'].transform(lambda x: x/x.sum())
        
        return df.sort_values(by=['year','month'])
    
    def backtest_model(self, df, weight):
        
        return df
        










        
        
        
        
        
        
        
        
        
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
        
    def __CheckLag(self, dt, freq): #last week or last month only
        if freq=='weekly':
            lim=15
        elif freq=='monthly':
            lim=40
        
        dt['dif'] = dt.groupby(['ticker'])['date'].diff()
        dt['dif'] = dt['dif'].fillna(pd.Timedelta(seconds=0)).dt.days.astype(int)
        dt['valid'] = dt['dif']<lim
        #dt.drop(['dif'], axis=1, inplace=True)
        
        return dt
        
        
        
        
        
        
        
        