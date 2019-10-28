
# coding: utf-8

# In[44]:


import os
import time
import datetime
import json
import gc
from numba import jit
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics

from itertools import product

import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
import altair as alt
alt.renderers.enable('notebook')
from altair.vega import v5
from IPython.display import HTML


# In[5]:


train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')
test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')
sub = pd.read_csv('sample_submission.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[6]:


train.shape


# In[7]:


test.shape


# In[8]:


train_identity.head()


# In[9]:


train_transaction.head()


# In[10]:


del train_identity, train_transaction, test_identity, test_transaction


# In[11]:


train.dtypes


# In[ ]:


#TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
#TransactionAMT: transaction payment amount in USD
#ProductCD: product code, the product for each transaction
#card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
#addr: address
#dist: distance
#P_ and (R__) emaildomain: purchaser and recipient email domain
#C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
#D1-D15: timedelta, such as days between previous transaction, etc.
#M1-M9: match, such as names on card and address, etc.
#Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
#Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. 
#They're collected by Vesta’s fraud protection system and digital security partners.
#(The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)
#DeviceType
#DeviceInfo
#id12 - id38


# In[57]:


obj_cols = train.dtypes
obj_cols[obj_cols=='object']


# In[12]:


train.describe()


# In[13]:


train.isnull().any().sum()


# In[14]:


# missing values
train_data_na = (train.isnull().sum() / len(train)) * 100
train_data_na =train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_data_na})
missing_data.head(20)


# In[15]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=train_data_na.index, y=train_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()


# In[16]:


#eda
# id_01 - id_11 are continuous variables, i
# d_12 - id_38 are categorical and the last two columns are obviously also categorical.
plt.hist(train['id_01'].dropna(), bins=77);
plt.title('Distribution of id_01 variable');
plt.show()


# In[60]:


plt.figure(figsize=(15,7))
plt.title('Distribution of Time Feature')
sns.distplot(train.TransactionDT)
plt.show()


# In[61]:


plt.figure(figsize=(15,7))
plt.title('Distribution of Time Feature')
sns.distplot(test.TransactionDT)
plt.show()


# In[62]:


plt.figure(figsize=(15,7))
plt.title('Distribution of Monetary Value Feature')
sns.distplot(train.TransactionAmt)
plt.show()


# In[63]:


plt.figure(figsize=(15,7))
sns.countplot(x='isFraud',data=train)
plt.title('CountPlot Frauds 1 = Positive , 0 = Negative')
plt.show()


# In[72]:


count = train.isFraud.value_counts()
regular = count[0]
frauds = count[1]
total= frauds + regular 
perc_reg = (float(regular)/total)*100
perc_frauds = (float(frauds)/total)*100
print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(regular, perc_reg, frauds, perc_frauds))


# In[17]:


train['id_01'].value_counts(dropna=False, normalize=True).head()


# In[18]:


train['id_03'].value_counts(dropna=False, normalize=True).head()


# In[19]:


#id_03 has 88% of missing values and 98% of values are either missing or equal to 0.


# In[20]:


plt.hist(train['id_07'].dropna(), bins=77);
plt.title('Distribution of id_07 variable');
plt.show()


# In[21]:


#Some of features seem to be normalized


# In[22]:


train['id_11'].value_counts(dropna=False, normalize=True).head()


# In[23]:


#22% of values in id_11 are equal to 100and 76% are missing.


# In[27]:


# categorical
fig, ax = plt.subplots(figsize = (30, 30))
plt.subplot(3, 4, 1)
sns.countplot(train['id_12']);
plt.subplot(3, 4, 2)
sns.countplot(train['id_15']);
plt.subplot(3, 4, 3)
sns.countplot(train['id_16']);
plt.subplot(3, 4, 4)
sns.countplot(train['id_23']);
plt.subplot(3, 4, 5)
sns.countplot(train['id_27']);
plt.subplot(3, 4, 6)
sns.countplot(train['id_28']);
plt.subplot(3, 4, 7)
sns.countplot(train['id_29']);
plt.subplot(3, 4, 8)
sns.countplot(train['id_34']);
plt.subplot(3, 4, 9)
sns.countplot(train['id_35']);
plt.subplot(3, 4, 10)
sns.countplot(train['id_36']);
plt.subplot(3, 4, 11)
sns.countplot(train['id_37']);
plt.subplot(3, 4, 12)
sns.countplot(train['id_38']);
plt.show()


# In[54]:


# client's device
fig, ax = plt.subplots(figsize = (60, 180))
plt.subplot(6, 1, 1)
sns.countplot(train['id_30']); # window 10, window 7
plt.xticks(rotation=90)
plt.subplot(6, 1, 2)
sns.countplot(train['id_31']); # chrome
plt.xticks(rotation=90)
plt.subplot(6, 1, 3)
sns.countplot(train['id_33']); # 1920*1080
plt.xticks(rotation=90)
plt.subplot(6, 1, 4)
sns.countplot(train['DeviceType']); # desktop mobile
plt.xticks(rotation=90)
plt.subplot(6, 1, 5)
sns.countplot(train['DeviceInfo']); # window mac ios
plt.xticks(rotation=90)
plt.show()


# In[31]:


plt.hist(train['TransactionDT'], label='train');
plt.hist(test['TransactionDT'], label='test');
plt.legend();
plt.title('Distribution of transactiond dates');
plt.show()


# In[ ]:


#A very important idea: it seems that train and test transaction dates don't overlap,
# so it would be prudent to use time-based split for validation


# In[56]:


fig, ax = plt.subplots(figsize = (30, 30))
plt.subplot(3, 4, 1)
sns.countplot(train['ProductCD']);
plt.subplot(3, 4, 2)
sns.countplot(train['card4']);
plt.subplot(3, 4, 3)
sns.countplot(train['card6']);
plt.subplot(3, 4, 4)
sns.countplot(train['M1']);
plt.subplot(3, 4, 5)
sns.countplot(train['M2']);
plt.subplot(3, 4, 6)
sns.countplot(train['M3']);
plt.subplot(3, 4, 7)
sns.countplot(train['M4']);
plt.subplot(3, 4, 8)
sns.countplot(train['M5']);
plt.subplot(3, 4, 9)
sns.countplot(train['M6']);
plt.subplot(3, 4, 10)
sns.countplot(train['M7']);
plt.subplot(3, 4, 11)
sns.countplot(train['M8']);
plt.subplot(3, 4, 12)
sns.countplot(train['M9']);
plt.show()


# In[ ]:


# card6 is type of card, card4 is credit card company


# In[55]:


fig, ax = plt.subplots(figsize = (60, 180))
plt.subplot(8, 1, 1)
sns.countplot(train['P_emaildomain']); # gmail.com hotmail.com yahoo.com
plt.xticks(rotation=90)
plt.subplot(8, 1, 2)
sns.countplot(train['R_emaildomain']); # gmail.com hotmail.com
plt.xticks(rotation=90)
plt.subplot(8, 1, 3)
sns.countplot(train['card1']); 
plt.xticks(rotation=90)
plt.subplot(8, 1, 4)
sns.countplot(train['card2']);
plt.xticks(rotation=90)
plt.subplot(8, 1, 5)
sns.countplot(train['card3']);
plt.xticks(rotation=90)
plt.subplot(8, 1, 6)
sns.countplot(train['card5']);
plt.xticks(rotation=90)
plt.subplot(8, 1, 7)
sns.countplot(train['addr1']);
plt.xticks(rotation=90)
plt.subplot(8, 1, 8)
sns.countplot(train['addr2']);
plt.xticks(rotation=90)
plt.show()


# In[74]:


# corr
corr = train.corr()
plt.figure(figsize=(15,7))
sns.heatmap(corr)
plt.title('Heatmap correlations Train_data')
plt.show()


# In[75]:


frauds = train[train['isFraud']==1]


# In[76]:


notfrauds= train[train['isFraud']==0]


# In[77]:


frauds.TransactionAmt.describe()


# In[78]:


notfrauds.TransactionAmt.describe()


# In[79]:


#plot of high value transactions
plt.figure(figsize=(15,7))
bins = np.linspace(200, 2500, 100)
plt.hist(notfrauds.TransactionAmt, bins, alpha=1, normed=True, label='Normal')
plt.hist(frauds.TransactionAmt, bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions \$200+)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)");
plt.show()


# In[ ]:


#Since the fraud cases are relatively few in number compared to bin size, 
#we see the data looks predictably more variable. In the long tail, especially, 
#we are likely observing only a single fraud transaction. 
#It would be hard to differentiate fraud from normal transactions by transaction amount alone.

