#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels.panel import compare
from statsmodels.formula.api import ols
from sklearn import datasets, linear_model, metrics 

data = pd.read_csv('US_data.csv')


# In[2]:


df = data.copy()

#set date
df['year'] = df.date.str[:4].astype(float)
df['date'] = pd.to_datetime(df['date']) 
df['inceptiondate'] = pd.to_datetime(df['inceptiondate'])

df = df.loc[df['securitytype']!='Open-End Fund'] #remove Open-end fund
df['age'] = np.log((df['date']-df['inceptiondate']).dt.days) #define age
df['lpd'] = df[['ticker','pd']].groupby('ticker').shift(1)
df['cd'] = df['pd']-df['lpd'] #change of discount
df['cd'] = df[['ticker','cd']].groupby('ticker').shift(-1) #lead change of discount 1 step for regression
df['cd'] = df['cd'].clip(lower=df['cd'].quantile(0.001), upper=df['cd'].quantile(0.999)) #windsorize outliers
lag = 4 #create AR lags
for l in range(lag):
    df['cdlag' + str(l+1)] = df[['ticker','cd']].groupby('ticker').shift(l+1)
    
np.warnings.filterwarnings('ignore')


# In[3]:


test = df.copy()

test['volume'] = test['volume']/1000000
test = test.loc[test['year'].isin(['2020','2018','2019'])]
test = test[['year','ticker','assetclasslevel1','assetclasslevel2','assetclasslevel3'
             ,'cd','cdlag1','pd','volume','age']]

test = test.dropna()


# In[16]:


test0 = test.set_index(['ticker','year'])

# fix assetclasslevel1, cluster time + ticker
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + age + assetclasslevel1',data = test0)
fit01 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix assetclasslevel2, cluster time + ticker
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + age + assetclasslevel2',data = test0)
fit02 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix assetclasslevel3, cluster time + ticker
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + age + assetclasslevel3',data = test0)
fit03 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix year, cluster time + ticker
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + TimeEffects',data = test0)
fit04 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# Compare
print(compare({'fixclass1':fit01
               ,'fixclass2':fit02
               ,'fixclass3':fit03
               ,'fixyear':fit04}))


# In[13]:


# fix time, cluster time + ticker
test1 = test.set_index(['ticker','year']) #entity and time multi-index
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + TimeEffects',data = test1)
fit1 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix time + ticker, cluster time + ticker
test2 = test.set_index(['ticker','year'])
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + EntityEffects + TimeEffects',data = test2)
fit2 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix assetclasslevel3, cluster time + assetclass3
test3 = test.set_index(['assetclasslevel3','year'])
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + EntityEffects',data = test3)
fit3 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix assetclasslevel1, cluster time + ticker
test4 = test.set_index(['ticker','year'])
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + assetclasslevel1',data = test4)
fit4 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# fix assetclasslevel3, cluster time + ticker
test5 = test.set_index(['ticker','year'])
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd + assetclasslevel3',data = test5)
fit5 = mod.fit(cov_type='clustered', cluster_time=True, cluster_entity=True)

# no fix
mod = PanelOLS.from_formula('cd ~ 1 + cdlag1 + volume + pd',data = test5)
fit6 = mod.fit(cov_type='robust')

# Compare
print(compare({'1':fit1
               ,'2':fit2
               ,'3':fit3
               ,'6':fit6}))


# In[209]:


print(fit5)


# In[212]:


# Use Statmodels
fit = ols('cd ~ cdlag1 + volume + pd + C(assetclasslevel1)', data=test).fit()
#test['pred1'] = fit.predict(test)
print(fit.summary())


# In[ ]:


EDITED2222223333333333


# In[ ]:


EDITED2222225555555555


# In[ ]:


EDITED333333


# In[ ]:


EDITED


# In[62]:


g = sns.relplot(x='cd', y='pred1', col='assetclasslevel1', hue='assetclasslevel1',
                kind="scatter", data=df)


# In[225]:


# fit = ols('cd ~ pd + volume', data=df).fit()
# df['pred'] = fit.predict(df[['pd','volume']])
# print(fit.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[214]:


### ACF (for each asset)

def acf_by_hand(inp, lag):
    x = inp.copy()
    x['cdlag'] = x[['ticker','cd']].groupby('cd').shift(lag)
    y2 = x.groupby('ticker', group_keys=False).apply(lambda x:x.iloc[lag:])['cd']
    y1 = x.groupby('ticker', group_keys=False).apply(lambda x:x.iloc[lag:])['cdlag']
    full = x['cd']
    sum_product = np.sum((y1-np.mean(full))*(y2-np.mean(full)))
    return sum_product / ((len(full) - lag) * np.var(full))

results = {}
nlags = 12
results["acf_by_hand"] = [acf_by_hand(x, lag+1) for lag in range(nlags)]
pd.DataFrame(results).plot(kind="bar", figsize=(10,5), grid=True)
plt.xlabel("lag")
plt.ylim([-1.2, 1.2])
plt.ylabel("ACF")
plt.show()

