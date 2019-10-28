
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import datetime 
import lightgbm as lgb

import os
import gc


# In[2]:


train=pd.read_csv('train_total.csv')
test=pd.read_csv('test_total.csv')


# In[3]:


train.shape


# In[4]:


X_train=pd.read_csv('train_fe4.csv')
X_test=pd.read_csv('test_fe4.csv')


# In[5]:


X_train.shape


# In[6]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[7]:


new_col=list(X_train_not_train)


# In[8]:


len(new_col)


# In[9]:


train[new_col]=X_train[new_col]


# In[10]:


train.shape


# In[11]:


test[new_col]=X_test[new_col]


# In[12]:


test.shape


# In[13]:


X_train=pd.read_csv('trian_xgb.csv')
X_test=pd.read_csv('test_xgb.csv')


# In[14]:


X_train.shape


# In[15]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[16]:


new_col=list(X_train_not_train)


# In[17]:


len(new_col)


# In[18]:


train[new_col]=X_train[new_col]


# In[19]:


train.shape


# In[20]:


test[new_col]=X_test[new_col]


# In[21]:


test.shape


# In[22]:


X_train=pd.read_csv('train_more.csv')
X_test=pd.read_csv('test_more.csv')


# In[23]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[24]:


new_col=list(X_train_not_train)


# In[25]:


len(new_col)


# In[26]:


train[new_col]=X_train[new_col]


# In[27]:


train.shape


# In[28]:


test[new_col]=X_test[new_col]


# In[29]:


test.shape


# In[30]:


X_train=pd.read_csv('train_exp.csv')
X_test=pd.read_csv('test_exp.csv')


# In[31]:


X_train.shape


# In[32]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[33]:


new_col=list(X_train_not_train)


# In[34]:


len(new_col)


# In[35]:


train[new_col]=X_train[new_col]


# In[36]:


train.shape


# In[37]:


test[new_col]=X_test[new_col]


# In[38]:


test.shape


# In[39]:


train.to_csv('train_final2.csv',index=False)
test.to_csv('test_final2.csv',index=False)


# In[40]:


train=pd.read_csv('train_final2.csv')
test=pd.read_csv('test_final2.csv')


# In[41]:


train.shape


# In[42]:


X_train=pd.read_csv('train_r2.csv')
X_test=pd.read_csv('test_r2.csv')


# In[43]:


X_train.shape


# In[44]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[45]:


new_col=list(X_train_not_train)


# In[46]:


len(new_col)


# In[47]:


train[new_col]=X_train[new_col]


# In[48]:


train.shape


# In[49]:


test[new_col]=X_test[new_col]


# In[50]:


test.shape


# In[5]:


train.to_csv('train_final3.csv',index=False)
test.to_csv('test_final3.csv',index=False)


# In[6]:


####打断点


# In[6]:


train=pd.read_csv('train_df98.csv')
test=pd.read_csv('test_df98.csv')


# In[7]:


train.shape


# In[1]:


X_train=pd.read_csv('train_final3.csv')
X_test=pd.read_csv('test_final3.csv')


# In[9]:


X_train.shape


# In[10]:


X_test.shape


# In[11]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[12]:


new_col=list(X_train_not_train)


# In[13]:


len(new_col)


# In[14]:


train[new_col]=X_train[new_col]


# In[15]:


train.shape


# In[19]:


test[new_col]=X_test[new_col]


# In[20]:


test.shape


# In[18]:


train.to_csv('train_final4.csv',index=False)
del train


# In[21]:


test.to_csv('test_final4.csv',index=False)


# In[22]:


test.head()


# In[9]:


test.all_columns


# In[3]:


####################


# In[4]:


train=pd.read_csv('train_final4.csv')
test=pd.read_csv('test_final4.csv')


# In[7]:


train["isFraud"].value_counts()


# In[8]:


test["isFraud"].value_counts()


# In[12]:


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
train.isna().sum()


# In[14]:


train["id_01"]


# In[ ]:




