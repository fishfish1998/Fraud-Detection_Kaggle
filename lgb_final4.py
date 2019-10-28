
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
get_ipython().magic('matplotlib inline')
import os, sys, gc, warnings, random, datetime


# In[2]:


train=pd.read_csv('train_final4.csv')
test=pd.read_csv('test_final4.csv')


# In[3]:


obj_cols = train.dtypes
obj_cols[obj_cols=='object']


# In[4]:


train['card2_and_count']=pd.to_numeric(train['card2_and_count'],errors='coerce')
train['addr1_card1']=pd.to_numeric(train['addr1_card1'],errors='coerce')


# In[5]:


test['card2_and_count']=pd.to_numeric(test['card2_and_count'],errors='coerce')
test['addr1_card1']=pd.to_numeric(test['addr1_card1'],errors='coerce')


# In[6]:


split_groups = train['DT_M']


# In[7]:


X= train.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
               'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)
y = train['isFraud']


# In[8]:


X.shape


# In[9]:


X_test = test.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
                    'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)


# In[10]:


X_test.shape


# In[11]:


del train, test
gc.collect()


# In[12]:


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


# In[13]:


SEED = 666
seed_everything(SEED)  
print(X.shape, X_test.shape)


# In[14]:


params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                } 


# In[15]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[16]:


columns = X.columns
splits = folds.split(X, y, groups=split_groups)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[17]:


feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[18]:


plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).tail(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[23]:


data=feature_importances.sort_values(by='average', ascending=False).tail(33) # 根据结果去设置
data


# In[24]:


sub = pd.read_csv('sample_submission.csv')
sub['isFraud'] = y_preds
sub.to_csv("lgb_final4.csv", index=False)#0.9482 


# In[25]:


no_contribution_feature=data['feature']


# In[26]:


no_contribution_feature


# In[27]:


X2=X.drop(no_contribution_feature, axis=1)


# In[28]:


X2.shape


# In[29]:


X_test2=X_test.drop(no_contribution_feature, axis=1)


# In[30]:


X_test2.shape


# In[31]:


del X, X_test
gc.collect()


# In[33]:


SEED = 666
seed_everything(SEED)  
print(X2.shape, X_test2.shape)


# In[34]:


params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                } 


# In[35]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[36]:


columns = X2.columns
splits = folds.split(X2, y, groups=split_groups)
y_preds = np.zeros(X_test2.shape[0])
y_oof = np.zeros(X2.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X2[columns].iloc[train_index], X2[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test2) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
# shuo
# Mean AUC = 0.9418500273607646
# Out of folds AUC = 0.9404989129711266


# In[40]:


sub = pd.read_csv('sample_submission.csv')
sub['isFraud'] = y_preds
sub.to_csv("lgb_final4_re.csv", index=False) #0.9489


# In[41]:


feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[42]:


plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).tail(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[44]:


data=feature_importances.sort_values(by='average', ascending=False).tail(3) #根据结果设置
data


# In[45]:


no_contribution_feature=data['feature']


# In[46]:


no_contribution_feature


# In[47]:


X3=X2.drop(no_contribution_feature, axis=1)


# In[48]:


X3.shape


# In[49]:


X_test3=X_test2.drop(no_contribution_feature, axis=1)


# In[50]:


X_test3.shape


# In[51]:


del X2, X_test2
gc.collect()


# In[52]:


SEED = 666
seed_everything(SEED)  
print(X3.shape, X_test3.shape)


# In[53]:


params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                } 


# In[54]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[55]:


columns = X3.columns
splits = folds.split(X3, y, groups=split_groups)
y_preds = np.zeros(X_test3.shape[0])
y_oof = np.zeros(X3.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X3[columns].iloc[train_index], X3[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test3) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[56]:


sub = pd.read_csv('sample_submission.csv')
sub['isFraud'] = y_preds
sub.to_csv("lgb_final4_re2.csv", index=False) # cv dropped, score dropped, 0.9485, no need to drop more.


# In[ ]:




