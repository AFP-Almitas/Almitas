"""
Created on Mon Dec 7 2020
@author: wanningd
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from cefpanelreg1 import CEFpanelreg
from datetime import datetime, date, timedelta

#filename = 'mergedWeekly.csv'
filename = 'US_data.csv'

"""
STEP3: Import data, train model, predict and compute SSE for each cross-validation sets
"""

# import data
data = pd.read_csv(filename)
data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])

# range of CD
data['lpd'] = data[['ticker', 'pd']].groupby('ticker').shift(1)
data['cd'] = data['pd'] - data['lpd']
range_cd = [np.nanmin(data['cd']), np.nanmax(data['cd'])]
print(range_cd)

cd = data['cd']
cd_filtered = cd[~np.isnan(cd)]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(cd_filtered)
plt.show()
# 25-75: [-0.385, 0.394]

fig = plt.figure(figsize=(10, 7))
plt.hist(cd_filtered, bins=100, range=[-1, 1])
plt.show()
