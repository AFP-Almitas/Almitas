# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:07:49 2020

@author: prhandojo
"""
import numpy as np
import pandas as pd

data = pd.read_csv('US_data.csv', index_col = 0)
data.columns = [x.lower() for x in data.columns]
data['date'] = pd.to_datetime(data['date'])
        
CEFA_data = pd.read_csv('CEFA_data.csv', index_col = 0)
CEFA_data['FileDate'] = pd.to_datetime(CEFA_data['FileDate'])

startdate = CEFA_data.FileDate.min()

# filter data to start from startdate
data = data.loc[data['date']>=startdate]

# check how many tickers are in both data set        
data_ticker = data.ticker.unique()
CEFA_ticker = CEFA_data.Ticker.unique()
inCEFA=np.zeros((len(data_ticker)))
for n in range(0,len(data_ticker)) :
    data_ticker[n] = data_ticker[n].split()[0]
    inCEFA[n] = data_ticker[n] in CEFA_ticker
sum(inCEFA)/len(inCEFA)

# remove the 'US' in all tickers in data
data['Ticker'] = data['ticker'].str.split(expand=True)[0]

# filter tickers that are not in CEFA
data = data.loc[data['Ticker'].isin(data_ticker[inCEFA==1])] 

# merged and retain all rows in data
merged = data.merge(CEFA_data, how='left', left_on=['Ticker', 'date'], right_on=['Ticker', 'FileDate'])

# remove ticker column and retain Ticker column
merged = merged.drop(columns = ['ticker', 'InceptionDate'])

# fill NA values using forward-filling method
merged['PctSharesOwnedbyInstitutions'] = merged.groupby(['Ticker'])['PctSharesOwnedbyInstitutions'].fillna(method='pad')

merged.to_csv (r'C:\Users\cauch\Documents\PC Docs\MFE\Courses\AFP\Almitas Data\Data\merged.csv', index = False, header=True)
