#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:24:03 2020

@author: kanp
"""

from cefpanelreg import CEFpanelreg

"""
STEP1: Input file name
"""

filename = 'merged2.csv'

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

dt = cef.data

cef.result(
        start_datetime = '2013-01-01',
        end_datetime = '2016-12-31',
        y = ['cd'],
        var_pit = [['cd',5]
                   ,['pd',5]
                   #,['expratio',5]
                   ],
        var_norm = [['cd',5,5,'mean']
                    #,['navchg',5,5,'mean']
                    #,['volume',5,5,'mean']
                    #,['totalyieldchg',1,10,'mean'],['navyieldchg',1,10,'mean']
                    #,['incyield',1,5,'mean']
                    ],
        fix = ['assetclasslevel3'],
        cluster = ['year','ticker']
        )

cef.summary()




dt.groupby(['assetclasslevel3'])['x'].count()
test=dt[dt.assetclasslevel3=='Mortgage']
