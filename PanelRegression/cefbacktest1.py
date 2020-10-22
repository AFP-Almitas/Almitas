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
                 method = ['mve',10,'2013-01-01','2017-12-31'],
                 freq = ['weekly'],
                 transcost = 0,
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
        

        

        
        df = self.data.copy()

        # filter weekly OR monthly
        if freq[0]=='weekly':
            #df['weekday'] = pd.to_datetime(df.date).dt.dayofweek
            #df = df.loc[df['weekday']==freq[1]]

            df = df.sort_values(by=['ticker','date']).groupby(['ticker','year','month','week']).first()
            df = df.reset_index()
            
            # weekly return            
            df['nextpriceclose'] = df.sort_values(['ticker','date']).groupby('ticker')['priceclose'].shift(-1)
            df['ret'] = df['nextpriceclose']/df['priceclose'] - 1
            df = df.drop(['nextpriceclose'],1)
            cols = list(df.columns)
            cols = cols[:5]+[cols[-1]]+cols[5:-1]
            df = df[cols]
            
            # check lag if it's one week interval
            df['dif'] = df.sort_values(['ticker','date']).groupby(['ticker','year'])['week'].diff().shift(-1)
            #df['dif'] = df['dif'].fillna(pd.Timedelta(seconds=0)).dt.days.astype(int)
            df = df[(df.dif==1)|(df.dif==0)|(((df.week==1)|(df.week==52))&((df.dif!=1)&(df.dif!=0)))] #sometimes last obs of the year dif is nan, 2nd from last is -51
            df.drop(['dif'], axis=1, inplace=True)
        
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
            df = df[(df.dif==1)|((np.isnan(df.dif))&(df.month==12))]
            df.drop(['dif'], axis=1, inplace=True)

        # change in discount
        if freq[0]=='weekly':
            ll = 1
            self.annualizecoef = 52
        elif freq[0]=='monthly':
            ll = 1
            self.annualizecoef = 12
        df['lpd'] = df[['ticker','pd']].groupby('ticker').shift(ll)
        df['cd'] = df['pd'] - df['lpd']


        # column reference
        c = len(df.columns)
        self.c = c
        
        # Point-in-time variables
        for var in alpha['pit']:
            variable,lag = var[0],var[1]
            df[variable+'_'+str(lag)] = df.groupby(['ticker'])[variable].shift(lag)
        
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
            func_dict[f](df,lag,length,variable)
        
        self.n_ind = len(alpha['pit']) + len(alpha['norm'])
            
        
        
        # filter asset class
        df = df.loc[df['assetclasslevel1'].isin(assetclass1)]

        # filter columns
        df = df[['ticker','year','month','week','date','assetclasslevel3','priceclose','ret','marketcap.x','cd'] + [col for col in df.columns[c:]]]
        self.c = 10
        print(df.columns.tolist())

        # filter dates
        df = df.loc[(df['date']>=start_datetime) & (df['date']<=end_datetime)]
        
        # save input
        self.input = df

        # backtest
        if method[0]=='scoring':
            self.result = self.backtest_scoring(df, method[1], method[2], method[3], freq[0], transcost)
            
        elif method[0]=='model':
            self.result = self.backtest_model(df)
            
        elif method[0]=='mve':
            self.result = self.backtest_mve(df, method[1], transcost, method[2], method[3], freq[0], method[4], method[5])

        return self.result
    
    def backtest_mve(self, df, nlookback, transcost, start_train, end_train, freq, portweight, nquintile):
        
        self.df0 = df
        # fit regression to training set
        fix = ['assetclasslevel3']
        fit = self.fitreg(df, start_train, end_train, ['cd'], [], [], fix, ['year','ticker'], self.c)
        print('b')
        # extract .nobs, .rsquared, .params, .tstats
        self.sumstat = {}
        self.sumstat['R2'] = round(fit.rsquared,4)
        self.sumstat['N'] = fit.nobs
        self.sumstat['Coefficient'] = round(fit.params,4)
        self.sumstat['t-stat'] = round(fit.tstats,4)
        
        # extract coefficients from fitted model
        self.coef = fit.params.reset_index()
        self.coefasset = self.coef['index'][0:-self.n_ind].tolist()
        
        # rename assetclasslevel3
        for i in range(1,len(self.coefasset)):
            self.coefasset[i] = self.coefasset[i][19:-1]
        
        self.assettrain = set(self.data['assetclasslevel3'].unique().tolist()) - {'Mortgage'} #
        self.set_coefasset = set(self.coefasset)
        self.assettest = set(df['assetclasslevel3'].unique().tolist()) - {'Mortgage'} #
        self.missingtrain = (self.assettrain - self.assettest)
        self.missingtest = (self.assettest - self.assettrain)
        
        self.interceptasset = list(self.assettrain - self.set_coefasset)
        self.coefasset[0] = self.interceptasset[0]
        self.coef['index'][0:-self.n_ind] = self.coefasset
        self.coef = self.coef.set_index('index')
        #print(self.coef)
        print('c')
        # manually create columns of 1 for each asset class
        fix_asfactors = pd.DataFrame(np.zeros((len(df), len(self.coefasset))))
        assetclasscol = df[fix].reset_index(drop=True)
        fix_asfactors = pd.concat([fix_asfactors, assetclasscol], axis=1)
        
        for i in range(1, len(self.coefasset)) : 
            index = fix_asfactors[fix]==self.coefasset[i]
            index = index.iloc[:,0] 
            fix_asfactors.loc[index,i] = 1
        
        # drop assetclasslevel3 column as it is no longer used
        fix_asfactors = fix_asfactors.drop(columns=fix)
        self.fix_asfactors = fix_asfactors
        
        # select columns of independent variables
        indeptvar = df.iloc[:,-self.n_ind:]
        self.indeptvar = indeptvar
        
        # construct the matrix of variables
        x = np.append(fix_asfactors, indeptvar, axis=1)
        x[:,0] = 1
        self.x = pd.DataFrame(x)
        
        # predict cd based on coefficients from model fitted on training data set
        pred = pd.DataFrame(np.dot(self.x, self.coef))
        self.pred = pred
        print('Asset class available in training but not in testing set:' + str(self.missingtrain))

        df = df.reset_index()
        df['pred'] = pred
    
    
        # keep date so all cefs in same week/month have the same date.
        if freq=='weekly':
            self.datekeep = df.sort_values('date').groupby(['year','month','week'])['date'].first()
        elif freq=='monthly':
            self.datekeep = df.sort_values('date').groupby(['year','month'])['date'].first()    
        df.drop(['date'], 1, inplace=True)
    
        if freq=='weekly':
            df = df.merge(self.datekeep, on=['year','month','week'])
            df['decile'] = df.groupby(['date'])['pred'].apply(pd.qcut, nquintile, labels=False)
            
        elif freq=='monthly':
            df = df.merge(self.datekeep, on=['year','month'])
            df['decile'] = df.groupby(['date'])['pred'].apply(pd.qcut, nquintile, labels=False)

        df = df.loc[(df['decile']==0) | (df['decile']==nquintile-1)]

        if portweight=='ew':
            if freq=='weekly':
                df['N'] = df.groupby(['year','month','week','decile'])['decile'].transform(lambda x: x.count())
            elif freq=='monthly':
                df['N'] = df.groupby(['year','month','decile'])['decile'].transform(lambda x: x.count())
            df['weight'] = 1/df['N']
        elif portweight=='vw':
            if freq=='weekly':
                df['N'] = df.groupby(['year','month','week','decile'])['decile'].transform(lambda x: x.count())
                df['weight'] = df.groupby(['year','month','week','decile'])['priceclose'].transform(lambda x: x/x.sum())
            elif freq=='monthly':
                df['N'] = df.groupby(['year','month','decile'])['decile'].transform(lambda x: x.count())
                df['weight'] = df.groupby(['year','month','decile'])['priceclose'].transform(lambda x: x/x.sum())
        
        
        print('start trans')
        # calculate transaction cost based on stock rebalancing
        self.d0 = df
        self.d1 = df[['date','decile','N','ticker','weight','ret']]
        self.cost = self.trans(self.d1, transcost)
        
        print('end trans')
    
        df =  df.sort_values(by=['year','month'])
        
        self.t0 = df
        df = df.assign(portret=df.weight*df.ret).groupby(['date','decile']).portret.sum().reset_index()
        df = df.pivot_table(index=['date'], columns=['decile'], values=['portret'])
        df.columns = ['short','long']

        for c in df.columns:
            df[c] = df[c].shift(1)
        df.iloc[0,] = 0
        
        df['longshort'] = df['long']-df['short']
        df['short'] = -df['short']
        
        # subtract transaction costs from returns
        df = df.merge(self.cost, on='date', how='left')
        df = df.fillna(0)
        for p in df.columns[1:4]:
            df[p] = df[p]-df[p+'cost']
        
        for l in df.columns[1:4]:
            df['cum_'+l] = df[l].transform(lambda x: (1+x).cumprod())
        
        df = df.set_index('date')
        

        self.port = df
        return self.port
        
        
    

    
    def trans(self, d1, bp):
        
        # weight
        self.d2 = d1.pivot_table(index=['date'], columns=['decile','ticker'], values=['weight'])
        self.d2.columns = self.d2.columns.droplevel(0)
        self.d2 = self.d2.fillna(0)

        # return
        self.d3 = d1.pivot_table(index=['date'], columns=['decile','ticker'], values=['ret'])
        self.d3.columns = self.d3.columns.droplevel(0)

        self.holding = d1.groupby(['date','decile'])['N'].first().reset_index().pivot_table(index=['date'], columns=['decile'], values=['N'])

        #self.short = self.d2.loc[:,([0])]
        #self.long = self.d2.loc[:,([9])]
        
        #self.shortchg, self.longchg = self.short.diff().abs(), self.long.diff().abs()
        #self.shortchg.iloc[0,], self.longchg.iloc[0,] = self.short.iloc[0,], self.long.iloc[0,]
        
        #self.longcost, self.shortcost = self.longchg.sum(1)*bp/10000, self.shortchg.sum(1)*bp/10000
        #self.longcost, self.shortcost = self.longcost.reset_index(), self.shortcost.reset_index()
        #cost = self.longcost.merge(self.shortcost, on='date')
        #cost.columns = ['date','longcost','shortcost']
        #cost['longshortcost'] = cost['longcost']+cost['shortcost']
        
        #cost.set_index('date')
        #return cost
        
        self.d2t = self.d2.copy()
        self.d2t.loc[:,([0])] = self.d2t.loc[:,([0])]*(-self.d3.loc[:,([0])])
        self.d2t.loc[:,([9])] = self.d2t.loc[:,([9])]*(self.d3.loc[:,([9])])
        self.d2t['shortret'] = self.d2t.loc[:,([0])].sum(1)
        self.d2t['longret'] = self.d2t.loc[:,([9])].sum(1)
        self.d2t['shortbal'] = self.d2t['shortret'].transform(lambda x: (1+x).cumprod())
        self.d2t['longbal'] = self.d2t['longret'].transform(lambda x: (1+x).cumprod())
        self.d2t = self.d2t.shift(1)
        self.d2t.iloc[0,-2:] = 1
        
        # value by stock before rebalancing
        self.d2b = self.d2.copy()
        self.d2b.loc[:,([0])] = self.d2b.loc[:,([0])].multiply(self.d2t['shortbal'], axis='index')
        self.d2b.loc[:,([9])] = self.d2b.loc[:,([9])].multiply(self.d2t['longbal'], axis='index')
        self.d2b = self.d2b*(1+self.d3)
        self.d2b = self.d2b.shift(1)
        
        # value after rebalancing
        self.d2.loc[:,([0])] = self.d2.loc[:,([0])].multiply(self.d2t['shortbal'], axis='index')
        self.d2.loc[:,([9])] = self.d2.loc[:,([9])].multiply(self.d2t['longbal'], axis='index')
        
        # cost
        self.dchg = (self.d2 - self.d2b).abs()
        self.shortcost = self.dchg.loc[:,([0])].sum(1).reset_index()
        self.longcost = self.dchg.loc[:,([9])].sum(1).reset_index()
        self.cost = self.longcost.merge(self.shortcost, on='date')
        self.cost.columns = ['date','longcost','shortcost']
        self.cost['longshortcost'] = self.cost['longcost']+self.cost['shortcost']
        self.cost.set_index('date')
        self.cost[['longcost','shortcost','longshortcost']] = self.cost[['longcost','shortcost','longshortcost']]*bp/10000
        
        return self.cost
    
    def plottest(self):
        
        plt.plot('cum_long',data=self.port)
        plt.plot('cum_short',data=self.port)
        plt.plot('cum_longshort',data=self.port)
        plt.legend()
        
        table = self.port.iloc[1:,0:3]
        table = table.agg([lambda x: x.mean()*self.annualizecoef,
                           lambda x: x.std()*np.sqrt(self.annualizecoef),
                           lambda x: x.mean()/x.std()*np.sqrt(self.annualizecoef)])
        table['stat'] = ['Avg','SD','Sharpe']
        table = table.set_index('stat')
        
        print(table)
        
        # store stats
        self.sharpels = table.iloc[2,2]
    
    def plotvsbm(self):
        
        
        
        
        pass
    
    def backtest_scoring(self, df, factorweight, portweight, nquintile, freq, transcost):

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

        # keep date so all cefs in same week/month have the same date.
        if freq=='weekly':
            self.datekeep = df.sort_values('date').groupby(['year','month','week'])['date'].first()
        elif freq=='monthly':
            self.datekeep = df.sort_values('date').groupby(['year','month'])['date'].first()    
        df.drop(['date'], 1, inplace=True)

        alpha_standard = df[df.columns[-k:]]
        df['score'] = alpha_standard.dot(w)

        df = df[df['score'].notna()]
        if freq=='weekly':
            df = df.merge(self.datekeep, on=['year','month','week'])
            df['decile'] = df.groupby(['date'])['score'].apply(pd.qcut, nquintile, labels=False)
            
        elif freq=='monthly':
            df = df.merge(self.datekeep, on=['year','month'])
            df['decile'] = df.groupby(['date'])['score'].apply(pd.qcut, nquintile, labels=False)

        df = df.loc[(df['decile']==0) | (df['decile']==nquintile-1)]

        if portweight=='ew':
            if freq=='weekly':
                df['N'] = df.groupby(['year','month','week','decile'])['decile'].transform(lambda x: x.count())
            elif freq=='monthly':
                df['N'] = df.groupby(['year','month','decile'])['decile'].transform(lambda x: x.count())
            df['weight'] = 1/df['N']
        elif portweight=='vw':
            if freq=='weekly':
                df['N'] = df.groupby(['year','month','week','decile'])['decile'].transform(lambda x: x.count())
                df['weight'] = df.groupby(['year','month','week','decile'])['marketcap.x'].transform(lambda x: x/x.sum())
            elif freq=='monthly':
                df['N'] = df.groupby(['year','month','decile'])['decile'].transform(lambda x: x.count())
                df['weight'] = df.groupby(['year','month','decile'])['marketcap.x'].transform(lambda x: x/x.sum())
        
        
        print('start trans')
        # calculate transaction cost based on stock rebalancing
        self.d0 = df
        self.d1 = df[['date','decile','N','ticker','weight','ret']]
        self.cost = self.trans(self.d1, transcost)
        
        print('end trans')
        
        df =  df.sort_values(by=['year','month'])
        
        self.t0 = df
        df = df.assign(portret=df.weight*df.ret).groupby(['date','decile']).portret.sum().reset_index()
        df = df.pivot_table(index=['date'], columns=['decile'], values=['portret'])
        df.columns = ['short','long']

        for c in df.columns:
            df[c] = df[c].shift(1)
        df.iloc[0,] = 0
        
        df['longshort'] = df['long']-df['short']
        df['short'] = -df['short']
        
        # subtract transaction costs from returns
        df = df.merge(self.cost, on='date', how='left')
        df = df.fillna(0)
        for p in df.columns[1:4]:
            df[p] = df[p]-df[p+'cost']
        
        for l in df.columns[1:4]:
            df['cum_'+l] = df[l].transform(lambda x: (1+x).cumprod())
        
        df = df.set_index('date')
        
        self.port = df
        return self.port
     
    def backtest_model(self, df, weight):
        
        
        
        self.port = df
        return self.port
        
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

        
        
        
        
        
        
        
        