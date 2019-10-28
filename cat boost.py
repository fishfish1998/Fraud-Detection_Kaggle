
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
from catboost import CatBoostClassifier, Pool, cv


# In[2]:


train=pd.read_csv('train_final4.csv')
test=pd.read_csv('test_final4.csv')


# In[3]:


train['card2_and_count']=pd.to_numeric(train['card2_and_count'],errors='coerce')
train['addr1_card1']=pd.to_numeric(train['addr1_card1'],errors='coerce')


# In[4]:


test['card2_and_count']=pd.to_numeric(test['card2_and_count'],errors='coerce')
test['addr1_card1']=pd.to_numeric(test['addr1_card1'],errors='coerce')


# In[5]:


split_groups = train['DT_M']


# In[6]:


X= train.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
               'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)
y = train['isFraud']


# In[7]:


X_test = test.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
                    'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)


# In[8]:


del train, test
gc.collect()


# In[9]:


X.fillna(-10000, inplace=True)
X_test.fillna(-10000, inplace=True)


# In[10]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[18]:


columns = X.columns
splits = folds.split(X, y, groups=split_groups)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_tr=X[columns].iloc[train_index]
    y_tr=y.iloc[train_index]
    X_val=X[columns].iloc[valid_index]
    y_val=y.iloc[valid_index]
    cat_params = {
        'iterations':5000,
        'learning_rate':0.02,
        'depth':9,
        'l2_leaf_reg':40,
        'bootstrap_type':'Bernoulli',
        'subsample':0.85,
        'loss_function': 'Logloss',
        'custom_loss':['AUC'],
        'logging_level':'Silent',
        #'task_type' : 'GPU',
        'early_stopping_rounds' : 100}
    
    model = CatBoostClassifier(**cat_params)
        
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),plot=True)

    y_pred_valid = model.predict(X_val)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_val, y_pred_valid)}")
    
    score += roc_auc_score(y_val, y_pred_valid) / NFOLDS
    y_preds += model.predict_proba(X_test)[:,1]/ NFOLDS
    
    del model,X_val,X_tr,y_val,y_tr
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[19]:


sub = pd.read_csv('sample_submission.csv')


# In[20]:


sub['isFraud'] = y_preds
sub.to_csv("cat_final4.csv", index=False) #0.9384


# In[11]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[12]:


splits = folds.split(X, y, groups=split_groups)


# In[13]:


for fold_n, (train_index, valid_index) in enumerate(splits):
    train_index_df=pd.DataFrame(train_index,columns=['train_index'])
    valid_index_df=pd.DataFrame(valid_index,columns=['valid_index'])
    del train_index, valid_index
    gc.collect()


# In[14]:


train_index_df.shape


# In[15]:


valid_index_df.shape


# In[16]:


from bayes_opt import BayesianOptimization


# In[17]:


train_index=train_index_df['train_index']
valid_index=valid_index_df['valid_index']


# In[18]:


X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]


# In[25]:


def Cat_bayesian(
    depth,
    l2_leaf_reg, 
    subsample,
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    l2_leaf_reg = int(l2_leaf_reg)
    depth = int(depth)

    assert type(l2_leaf_reg) == int
    assert type(depth) == int
    y_oof = np.zeros(X_valid.shape[0])

    param = { 'iterations':10000,
              'learning_rate':0.01,
              'subsample' : subsample,
              'l2_leaf_reg' : l2_leaf_reg,
              'depth': depth,
              'bootstrap_type':'Bernoulli',
              'loss_function': 'Logloss',
              'custom_loss':['AUC'],
              'logging_level':'Silent',
               #'task_type' : 'GPU',
              'early_stopping_rounds' : 100}    
    
    model = CatBoostClassifier(**param)
    
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid),verbose=False, use_best_model=True)
        
    y_oof = model.predict(X_valid)

    score = roc_auc_score(y_valid, y_oof)

    return score


# In[26]:


bounds_Cat = {
    'l2_leaf_reg': (20, 400), 
    'subsample' : (0.2,0.9),
    'depth':(-1,15),
}


# In[27]:


Cat_BO = BayesianOptimization(Cat_bayesian, bounds_Cat)


# In[28]:


print(Cat_BO.space.keys)


# In[29]:


init_points = 3
n_iter = 7


# In[ ]:


print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    Cat_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[13]:


params = {'iterations':10000,
          'learning_rate':0.01,
          'depth':15,
          'l2_leaf_reg':20,
          'bootstrap_type':'Bernoulli',
          'subsample':0.7211,
          'loss_function': 'Logloss',
          'custom_loss':['AUC'],
          'logging_level':'Silent',
          #'task_type' : 'GPU',
          'early_stopping_rounds' : 100} # copy this one


# In[14]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[15]:


columns = X.columns
splits = folds.split(X, y, groups=split_groups)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importance_df = pd.DataFrame()

  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_tr=X[columns].iloc[train_index]
    y_tr=y.iloc[train_index]
    X_val=X[columns].iloc[valid_index]
    y_val=y.iloc[valid_index]
    
    model = CatBoostClassifier(**params)
        
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),plot=True)

    y_pred_valid = model.predict(X_val)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_val, y_pred_valid)}")
    
    score += roc_auc_score(y_val, y_pred_valid) / NFOLDS
    y_preds += model.predict_proba(X_test)[:,1]/ NFOLDS
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df['feature']=columns
    fold_importance_df['importance']=model.get_feature_importance()
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del model,X_val,X_tr,y_val,y_tr
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[17]:


sub = pd.read_csv('sample_submission.csv')


# In[18]:


sub['isFraud'] = y_preds
sub.to_csv("cat_final4_bayes.csv", index=False) #0.9445


# In[ ]:




