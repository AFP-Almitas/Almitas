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

filename = 'US_data.csv'

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
        start_datetime = '2019-02-01',
        end_datetime = '2020-4-1',
        y = ['cd'],
        var_pit = [['cd',1]],
        var_norm = [['volume',1,10,'std'],['nav',1,10,'mean'],['cd',1,10,'mean'],['cd',1,10,'std']],
        fix = ['assetclasslevel3'],
        cluster = ['year','ticker']
        )
cef.summary()


#table = cef.data
#test: 2013-2016
#try premium/discount group seperately

"""

# view specific columns
cef(['ticker','volume'])

"""