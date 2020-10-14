#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:28:49 2020

@author: kanp
"""

import pandas as pd


data = pd.read_csv('merged.csv', index_col = None)
data
data.columns = [x.lower() for x in data.columns]


#### Imputation:fill NA values using forward-filling method

imputelist = ['totalyield','navyield','capgainpct','incyield','nav',
              'unratedbondspct','levadjnavyield','earningsyield']
for i in imputelist:
    data[i] = data.groupby(['ticker'])[i].fillna(method='pad')

#data['unratedbondspct'].isna().sum()

#data.pop('ticker')
#data.insert(0, 'ticker', first_col)


#### New Variables:difference

difflist = ['totalyield','navyield','capgainpct','incyield','nav',
            'unratedbondspct','levadjnavyield','earningsyield']
for i in difflist:
    data[i+'chg'] = data.groupby(['ticker'])[i].diff()


####


#### Save

data.dtypes
data.to_csv('merged2.csv')
