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
method 2) ['model']
method 3) ['mve',lookback period,starttrain,endtrain,'ew' or 'vw',#quintile] 
freq: ['weekly'] or ['monthly',0 beginning of month or 1 end of month]
"""

############### weekly #####################

cef = CEFbacktest(filename)
data = cef.data

dt = cef.backtest(
    start_datetime = '2017-01-01',
    end_datetime = '2017-12-31',
    assetclass1 = ['Equity','Fixed Income','Commodity'],
    alpha = {'pit':[['cd',5],['pd',5],['navchg',5]],
             'norm':[['cd',5,10,'mean'],['navchg',5,10,'mean']]},
    method = ['mve',10,'2013-01-01','2016-12-31','vw',10],
    #method = ['scoring',{'pit':[],'norm':[0.8,0.2]},'ew',10],
    freq = ['weekly'],
    transcost = 1,
    compare = False)

cef.plottest()


############### monthly ####################

cef = CEFbacktest(filename)
data = cef.data


dt = cef.backtest(
    start_datetime = '2017-01-01',
    end_datetime = '2018-12-31',
    assetclass1 = ['Equity','Fixed Income','Commodity'],
    alpha = {'pit':[['cd',22],['pd',22],['navchg',22]],
             'norm':[['cd',22,10,'mean'],['navchg',5,10,'mean']]},
    method = ['mve',10,'2013-01-01','2016-12-31','ew',10],
    #method = ['scoring',{'pit':[],'norm':[0.8,0.2]},'ew',10],
    freq = ['monthly',0],
    transcost = 20,
    compare = False)

cef.plottest()










cef.data
cef.c
inp = cef.input
coef = cef.coef
coefasset = cef.coefasset
assettrain = cef.assettrain
misstrain = cef.missingtrain
interceptasset = cef.interceptasset
indeptvar = cef.indeptvar
df = cef.df0
fix_asfactors = cef.fix_asfactors
x = cef.x
port = cef.port
pred = cef.pred

df = cef.d0
d2 = cef.d2
d2t = cef.d2t
d2b = cef.d2b
d3 = cef.d3


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd










port = cef.port
holding = cef.holding
cost = cef.cost
inp = cef.input
d1 = cef.d1
