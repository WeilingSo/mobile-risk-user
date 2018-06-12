
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# In[2]:


train = pd.read_csv('../data/train_featureV1.csv')
test = pd.read_csv('../data/test_featureV1.csv')


# In[3]:


train.head()


# In[48]:


dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))


# In[45]:


def evalMetric(preds,dtrain):
    
    label = dtrain.get_label()
    
    
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    
    auc = metrics.roc_auc_score(pre.label,pre.preds)

    pre.preds=pre.preds.map(lambda x: 1 if x>=0.38 else 0)

    f1 = metrics.f1_score(pre.label,pre.preds)
    
    
    res = 0.6*auc +0.4*f1
    
    return 'res',res,True
    

    


# ## lgb

# In[46]:


lgb_params =  {
    'seed':0,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'is_training_metric': False,
    'learning_rate': 0.02,
    'verbosity':-1,
    'colsample_bytree':0.7,
}

lgb_params1 = lgb_params.copy()
lgb_params1['seed'] = 1

lgb_params2 = lgb_params.copy()
lgb_params2['seed'] = 2

lgb_params3 = lgb_params.copy()
lgb_params3['seed'] = 3

lgb_params4 = lgb_params.copy()
lgb_params4['seed'] = 4

lgb_params5 = lgb_params.copy()
lgb_params5['seed'] = 5

lgb_params6 = lgb_params.copy()
lgb_params6['seed'] = 6

lgb_params7 = lgb_params.copy()
lgb_params7['seed'] = 7

lgb_params8 = lgb_params.copy()
lgb_params8['seed'] = 8

lgb_params9 = lgb_params.copy()
lgb_params9['seed'] = 9



# ### 本地CV

# In[7]:


lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# ## 训练

# In[8]:


model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=392,valid_sets=[dtrain])


# In[9]:


lgb.cv(lgb_params1,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[10]:


model1 =lgb.train(lgb_params1,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=372,valid_sets=[dtrain])


# In[11]:


lgb.cv(lgb_params2,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[14]:


model2 =lgb.train(lgb_params3,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=322,valid_sets=[dtrain])


# In[13]:


lgb.cv(lgb_params3,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[16]:


model3 =lgb.train(lgb_params3,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=324,valid_sets=[dtrain])


# In[15]:


lgb.cv(lgb_params4,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[18]:


model4 =lgb.train(lgb_params4,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=334,valid_sets=[dtrain])


# In[17]:


lgb.cv(lgb_params5,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[20]:


model5 =lgb.train(lgb_params5,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=352,valid_sets=[dtrain])


# In[19]:


lgb.cv(lgb_params6,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[22]:


model6 =lgb.train(lgb_params6,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=212,valid_sets=[dtrain])


# In[21]:


lgb.cv(lgb_params7,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[24]:


model7 =lgb.train(lgb_params7,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=322,valid_sets=[dtrain])


# In[23]:


lgb.cv(lgb_params8,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[26]:


model8 =lgb.train(lgb_params8,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=174,valid_sets=[dtrain])


# In[25]:


lgb.cv(lgb_params9,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=2,num_boost_round=500,nfold=3,metrics=['evalMetric'])


# In[49]:


model9 =lgb.train(lgb_params9,dtrain,feval=evalMetric,verbose_eval=100,num_boost_round=420,valid_sets=[dtrain])


# ## xgb

# In[55]:


xgb_params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'stratified':True,
    #'max_depth':10,
    # 'gamma':1,
    #'subsample':0.8,
    #'colsample_bytree':0.8,
    # 'lambda':1,
    'eta':0.01,
    'seed':20,
    'silent':1,
}

xgb_params2 = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'stratified':True,
    #'max_depth':10,
    # 'gamma':1,
    #'subsample':0.8,
    #'colsample_bytree':0.8,
    # 'lambda':1,
    'eta':0.01,
    'seed':6,
    'silent':1,
}


def evalMetric(preds,dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds=pre.preds.map(lambda x: 1 if x>=0.38 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc +0.4*f1
    return 'res',res


# In[28]:


dtrain = xgb.DMatrix(train.drop(['uid','label'],axis=1),label=train.label)
xgb.cv(xgb_params,dtrain,num_boost_round=700,nfold=3,verbose_eval=10,early_stopping_rounds=100,maximize=True,feval=evalMetric)


# In[57]:


dtrain = xgb.DMatrix(train.drop(['uid','label'],axis=1),label=train.label)
xgbmodel=xgb.train(xgb_params,dtrain=dtrain,num_boost_round=620,verbose_eval=100,
                evals=[(dtrain,'train')],maximize=True,feval=evalMetric,early_stopping_rounds=100)


# In[30]:


xgb.cv(xgb_params2,dtrain,num_boost_round=700,nfold=3,verbose_eval=10,early_stopping_rounds=100,maximize=True,feval=evalMetric)


# In[34]:


dtrain = xgb.DMatrix(train.drop(['uid','label'],axis=1),label=train.label)
xgbmode2=xgb.train(xgb_params2,dtrain=dtrain,num_boost_round=670,verbose_eval=100,
                evals=[(dtrain,'train')],maximize=True,feval=evalMetric,early_stopping_rounds=100)


# ### 预测

# In[35]:


pred0=model.predict(test.drop(['uid'],axis=1))


# In[36]:


pred1=model1.predict(test.drop(['uid'],axis=1))


# In[37]:


pred2=model2.predict(test.drop(['uid'],axis=1))


# In[38]:


pred3=model3.predict(test.drop(['uid'],axis=1))


# In[39]:


pred4=model4.predict(test.drop(['uid'],axis=1))


# In[40]:


pred5=model5.predict(test.drop(['uid'],axis=1))


# In[41]:


pred6=model6.predict(test.drop(['uid'],axis=1))


# In[42]:


pred7=model7.predict(test.drop(['uid'],axis=1))


# In[43]:


pred8=model8.predict(test.drop(['uid'],axis=1))


# In[50]:


pred9=model9.predict(test.drop(['uid'],axis=1))


# In[58]:


dtest = xgb.DMatrix(test.drop(['uid'],axis=1))
xgbpreds =xgbmodel.predict(dtest)
xgbpreds2 =xgbmode2.predict(dtest)


# In[59]:


pred =pd.DataFrame({'p0':pred0,'p1':pred1,'p2':pred2,'p3':pred3,'p4':pred4,'p5':pred5,'p6':pred6,'p7':pred7,'p8':pred8,'p9':pred9,'pxgb1':xgbpreds,'xgbp2':xgbpreds2})
pred.head()


# In[60]:


# 求平均值
pred_t = pred.T
pred_mean = pred_t.mean()
res =pd.DataFrame({'uid':test.uid,'label':pred_mean})


# In[61]:


res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.38 else 0)
#res.label = res.label.map(lambda x: int(x))


# In[62]:


res.to_csv('../result/lgb-baseline.csv',index=False,header=False,sep=',',columns=['uid','label'])

