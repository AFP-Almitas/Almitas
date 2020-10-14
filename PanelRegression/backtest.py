#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:56:49 2020

@author: kanp
"""

from cefbacktest import CEFbacktest

"""
STEP1: Input file name
"""

filename = 'merged2.csv'

"""
STEP2: Input parameters for backtest
method 1) ['scoring',factor weight,'ew' or 'vw',#quintile]
method 2) []
method 3) []
"""

cef = CEFbacktest(filename)

dt = cef.data

dt2 = cef.backtest(
    start_datetime = '2013-01-01',
    end_datetime = '2016-12-31',
    assetclass1 = 'Equity',
    alpha = {'pit':[],
             'norm':[['cd',5,5,'mean'],['volume',5,5,'mean']]},
    method = ['scoring', #'scoring' or 'model' or 'mve'
              {'pit':[],'norm':[0.5,0.5]},
              'vw',
              10], 
    freq = ['weekly'], #['weekly'] or ['monthly',0 beginning of month or 1 end of month]
    compare = False)



dt3=dt2.groupby(['year','week'])['date'].transform(lambda x: len(x.unique()))
dt4=dt2.groupby(['year','week']).agg({'ndate':'sum'})

         
        alpha_standard = dt2[dt2.columns[-2:]]
        
        
        print(alpha_standard.dot(w))
        





dt2

result = cef.backtest(
    start_datetime = '2013-01-01',
    end_datetime = '2016-12-31')


cef
