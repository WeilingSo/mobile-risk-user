
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[3]:


voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[4]:


prefix = np.array(['u'])
uid_num = np.arange(7000,10000)
uid_num_char = uid_num.astype('U')
uid_num_str = np.core.defchararray.add(prefix, uid_num_char)
uid_test = pd.DataFrame(uid_num_str, columns=['uid'])
uid_test.to_csv('../data/uid_test_a.txt',index=None)


# In[5]:


voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)


# In[6]:


voice.start_time = voice.start_time.astype(int)
voice.end_time = voice.end_time.astype(int)

voice['date'] = voice.start_time//1000000
voice['hour'] = voice.start_time%1000000//10000
voice['time'] = voice.start_time//100%10000
voice['voice_dura']=abs(voice.end_time.astype('int')-voice.start_time.astype('int'))

def map_dure_to_cata(x):
    if (x < 2):
        return 1
    elif (x >= 2 & x < 5):
        return 2
    else:
        return 3

voice['dura_cls'] = voice['voice_dura'].apply(map_dure_to_cata)

sms.start_time = sms.start_time.astype(int)
sms['date'] = sms.start_time//1000000
sms['hour'] = sms.start_time%1000000//10000

wa.date = wa.date.fillna(0).astype(int)
wa.up_flow = wa.up_flow.fillna(0).astype(int)
wa.down_flow = wa.down_flow.fillna(0).astype(int)
wa.visit_dura = wa.visit_dura.fillna(0).astype(int)
wa.visit_cnt = wa.visit_cnt.fillna(0).astype(int)


# In[7]:


def map_dure_to_cata(x):
    if (x < 2):
        return 1
    elif (x >= 2 & x < 10):
        return 2
    else:
        return 3

voice['dura_cls'] = voice['voice_dura'].apply(map_dure_to_cata)


# ### 通话记录

# In[8]:


# 不同的电话号码数/电话总数
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

# 不同的电话号码头三位的数量
voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

# 每种电话长度的通话次数
voice_opp_len1=voice[voice.in_out==1].groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice_opp_len0=voice[voice.in_out==0].groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice_opp_len = pd.merge(voice_opp_len1,voice_opp_len0,how='outer',on='uid').fillna(0)

voice_dura_opp_len =voice.groupby(['uid','opp_len'])['voice_dura'].sum().unstack().add_prefix('voice_dura_opp_len_').reset_index().fillna(0)

# 每种类型通话的次数
voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

# 每种类型通话的平均时长
voice_dura_type = voice.groupby(['uid','call_type'])['voice_dura'].mean().unstack().add_prefix('voice_dura_type_').reset_index().fillna(0)

# 接入/打出的电话总数
voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

# 通话时长的各统计量
voice_dura = voice.groupby(['uid'])['voice_dura'].agg(['std','max','median','mean','sum']).add_prefix('voice_dura_').reset_index().fillna(0)

## 每个用户收/发电话的号码的不同号码数
voice_opp_len_inout = voice.groupby(['uid','in_out'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).unstack().add_prefix('voice_opp_inout_num_').reset_index().fillna(0)

# 不同的日期数
voice_day_count = voice.groupby(['uid'])['date'].agg({'voice_day_count': lambda x: len(pd.unique(x))}).reset_index().fillna(0)

# 每天in/out电话量
voice_everyday_in_count = voice[voice.in_out==1].groupby(['uid','date'])['uid'].count().unstack().add_prefix('voice_everyday_in_count').reset_index().fillna(0)
voice_everyday_out_count = voice[voice.in_out==0].groupby(['uid','date'])['uid'].count().unstack().add_prefix('voice_everyday_out_count').reset_index().fillna(0)

voice_everyday_in_dura = voice[voice.in_out==1].groupby(['uid','date'])['voice_dura'].sum().unstack().add_prefix('voice_everyday_in_dura').reset_index().fillna(0)
voice_everyday_out_dura = voice[voice.in_out==0].groupby(['uid','date'])['voice_dura'].sum().unstack().add_prefix('voice_everyday_out_dura').reset_index().fillna(0)

# 每个小时段的平均电话量
voice_hour_count = voice.groupby(['uid','hour'])['voice_dura'].sum().unstack().add_prefix('voice_hour_count').reset_index().fillna(0)


# In[9]:


voice_everyday_call_type = voice.groupby(['uid','date','call_type'])['uid'].count().unstack().add_prefix('voice_everyday_call_type_').reset_index().fillna(0)
voice_everyday_call_type_1 = voice_everyday_call_type.groupby(['uid','date'])['voice_everyday_call_type_1'].sum().unstack().add_prefix('day_').reset_index().fillna(0)
voice_everyday_call_type_2 = voice_everyday_call_type.groupby(['uid','date'])['voice_everyday_call_type_2'].sum().unstack().add_prefix('day_').reset_index().fillna(0)
voice_everyday_call_type_3 = voice_everyday_call_type.groupby(['uid','date'])['voice_everyday_call_type_3'].sum().unstack().add_prefix('day_').reset_index().fillna(0)
voice_everyday_call_type_4 = voice_everyday_call_type.groupby(['uid','date'])['voice_everyday_call_type_4'].sum().unstack().add_prefix('day_').reset_index().fillna(0)
voice_everyday_call_type_5 = voice_everyday_call_type.groupby(['uid','date'])['voice_everyday_call_type_5'].sum().unstack().add_prefix('day_').reset_index().fillna(0)
voice_everyday_call_type = pd.merge(voice_everyday_call_type_1,voice_everyday_call_type_2,how='left',on='uid')
voice_everyday_call_type = pd.merge(voice_everyday_call_type,voice_everyday_call_type_3,how='left',on='uid')
voice_everyday_call_type = pd.merge(voice_everyday_call_type,voice_everyday_call_type_4,how='left',on='uid')
voice_everyday_call_type = pd.merge(voice_everyday_call_type,voice_everyday_call_type_5,how='left',on='uid')


# In[10]:


# 每种类型通话的平均时长
voice_dura_cls_type = voice.groupby(['uid','call_type'])['dura_cls'].mean().unstack().add_prefix('voice_dura_cls_type_').reset_index().fillna(0)

voice_dura_cls0 = voice[voice.in_out==0].groupby(['uid','dura_cls'])['uid'].count().unstack().add_prefix('voice_out_dura_cls_').reset_index().fillna(0)
voice_dura_cls1 = voice[voice.in_out==1].groupby(['uid','dura_cls'])['uid'].count().unstack().add_prefix('voice_in_dura_cls_').reset_index().fillna(0)
voice_dura_cls = pd.merge(voice_dura_cls0,voice_dura_cls1,how='outer',on='uid').fillna(0)


# In[11]:


# 每种电话类型每种通话时长分类的总通话时长
voice_everyday_dura_type = voice.groupby(['uid','dura_cls','call_type'])['uid'].count().unstack().add_prefix('voice_everyday_call_type_').reset_index().fillna(0)
voice_everyday_dura_type_1 = voice_everyday_dura_type.groupby(['uid','dura_cls'])['voice_everyday_call_type_1'].sum().unstack().add_prefix('dura_cls_').add_prefix('type1_').reset_index().fillna(0)
voice_everyday_dura_type_2 = voice_everyday_dura_type.groupby(['uid','dura_cls'])['voice_everyday_call_type_2'].sum().unstack().add_prefix('dura_cls_').add_prefix('type2_').reset_index().fillna(0)
voice_everyday_dura_type_3 = voice_everyday_dura_type.groupby(['uid','dura_cls'])['voice_everyday_call_type_3'].sum().unstack().add_prefix('dura_cls_').add_prefix('type3_').reset_index().fillna(0)
voice_everyday_dura_type_4 = voice_everyday_dura_type.groupby(['uid','dura_cls'])['voice_everyday_call_type_4'].sum().unstack().add_prefix('dura_cls_').add_prefix('type4_').reset_index().fillna(0)
voice_everyday_dura_type_5 = voice_everyday_dura_type.groupby(['uid','dura_cls'])['voice_everyday_call_type_5'].sum().unstack().add_prefix('dura_cls_').add_prefix('type5_').reset_index().fillna(0)
voice_everyday_dura_type = pd.merge(voice_everyday_dura_type_1,voice_everyday_dura_type_2,how='left',on='uid')
voice_everyday_dura_type = pd.merge(voice_everyday_dura_type,voice_everyday_dura_type_3,how='left',on='uid')
voice_everyday_dura_type = pd.merge(voice_everyday_dura_type,voice_everyday_dura_type_4,how='left',on='uid')
voice_everyday_dura_type = pd.merge(voice_everyday_dura_type,voice_everyday_dura_type_5,how='left',on='uid')
#voice_everyday_dura_type.head()


# In[12]:


# 每种类型通话的次数
voice_opp_head_count = voice.groupby(['uid','opp_head'])['voice_dura'].count().unstack().add_prefix('voice_opp_head_count_').reset_index().fillna(0)
voice_opp_head_count.head()


# In[13]:


# 前k个频繁被拨打的号码，每个用户拨打的时间
voice_opp_num_first_k_ori = voice['opp_head'].value_counts()[0:30]
voice_opp_num_first_k = []
for i in np.arange(0,voice_opp_num_first_k_ori.shape[0]):
    voice_opp_num_first_k.append(voice_opp_num_first_k_ori[i:i+1].to_string().split(' ')[0])

voice_num_else = voice[~voice.opp_head.isin(voice_opp_num_first_k)]
voice_num_first_k_dura = voice_num_else.groupby(['uid'])['voice_dura'].count().reset_index().fillna(0).rename(columns={'uid': 'uid', 'voice_dura': 'voice_num_else_dura'})
    
for num in voice_opp_num_first_k:
    voice_num_k_dura = voice[voice.opp_head == num].groupby(['uid'])['voice_dura'].count().reset_index().fillna(0)
    voice_num_first_k_dura = pd.merge(voice_num_first_k_dura,voice_num_k_dura,how='outer',on='uid')

voice_num_first_k_dura.fillna(0,inplace=True)


# In[14]:


# 通话时间的各统计量
voice_time = voice.groupby(['uid'])['time'].agg(['std','median','mean']).add_prefix('voice_time_').reset_index().fillna(0)


# ## 短信记录

# In[15]:


# 不同的in/out短信号码数/电话总数
sms_out_opp_num = sms[sms.in_out==0].groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_out_opp_num').reset_index()
sms_in_opp_num = sms[sms.in_out==1].groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_in_opp_num').reset_index()

# in/out号码不同头三位的数量
sms_opp_head=sms.groupby(['uid','in_out'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).unstack().add_prefix('sms_opp_head_').reset_index()

# 每种号码长度的短信次数
sms_out_opp_len=sms[sms.in_out==0].groupby(['uid','opp_len'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).unstack().add_prefix('sms_out_opp_len').reset_index().fillna(0)
sms_in_opp_len=sms[sms.in_out==1].groupby(['uid','opp_len'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).unstack().add_prefix('sms_in_opp_len').reset_index().fillna(0)

# 接受/发出短信总数
sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)

# 不同的日期数
sms_day_count = sms.groupby(['uid'])['start_time'].agg({'sms_day_count': lambda x: len(pd.unique(x//1000000))}).reset_index().fillna(0)

# 每天in/out短信量
sms_everyday_out_count = sms[sms.in_out==0].groupby(['uid','date'])['uid'].count().unstack().add_prefix('sms_everyday_out_count').reset_index().fillna(0)
sms_everyday_in_count = sms[sms.in_out==1].groupby(['uid','date'])['uid'].count().unstack().add_prefix('sms_everyday_in_count').reset_index().fillna(0)

# 每个小时段的平均in/out短信量
sms_hour_out_count = sms[sms.in_out==0].groupby(['uid','hour'])['uid'].count().unstack().add_prefix('sms_hour_out_count').reset_index().fillna(0)
sms_hour_in_count = sms[sms.in_out==1].groupby(['uid','hour'])['uid'].count().unstack().add_prefix('sms_hour_in_count').reset_index().fillna(0)


# In[16]:


# 每5天in/out不同的对端号码
sms_everyday_out_opp_head_unique = sms[sms.in_out==0].groupby(['uid','date'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).unstack().add_prefix('sms_everyday_out_opp_head_unique').reset_index().fillna(0)
sms_everyday_in_opp_head_unique = sms[sms.in_out==1].groupby(['uid','date'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).unstack().add_prefix('sms_everyday_in_opp_head_unique').reset_index().fillna(0)

out_opp_head_unique = sms_everyday_out_opp_head_unique.drop(['uid'],axis=1)
out_opp_head_unique.columns=np.arange(1,46)
sms_every5day_out_opp_head_unique = pd.DataFrame()
sms_every5day_out_opp_head_unique['uid'] = sms_everyday_out_opp_head_unique.uid
sms_every5day_out_opp_head_unique['1'] = out_opp_head_unique.loc[:,1:5].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['2'] = out_opp_head_unique.loc[:,6:10].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['3'] = out_opp_head_unique.loc[:,11:15].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['4'] = out_opp_head_unique.loc[:,16:20].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['5'] = out_opp_head_unique.loc[:,21:25].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['6'] = out_opp_head_unique.loc[:,26:30].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['7'] = out_opp_head_unique.loc[:,31:35].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['8'] = out_opp_head_unique.loc[:,36:40].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_unique['9'] = out_opp_head_unique.loc[:,41:45].apply(lambda x: x.sum(), axis=1)

in_opp_head_unique = sms_everyday_in_opp_head_unique.drop(['uid'],axis=1)
in_opp_head_unique.columns=np.arange(1,46)
sms_every5day_in_opp_head_unique = pd.DataFrame()
sms_every5day_in_opp_head_unique['uid'] = sms_everyday_in_opp_head_unique.uid
sms_every5day_in_opp_head_unique['1'] = in_opp_head_unique.loc[:,1:5].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['2'] = in_opp_head_unique.loc[:,6:10].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['3'] = in_opp_head_unique.loc[:,11:15].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['4'] = in_opp_head_unique.loc[:,16:20].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['5'] = in_opp_head_unique.loc[:,21:25].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['6'] = in_opp_head_unique.loc[:,26:30].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['7'] = in_opp_head_unique.loc[:,31:35].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['8'] = in_opp_head_unique.loc[:,36:40].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_unique['9'] = in_opp_head_unique.loc[:,41:45].apply(lambda x: x.sum(), axis=1)


# In[17]:


# 每5天in/out的电话数量
sms_everyday_out_opp_head_count = sms[sms.in_out==0].groupby(['uid','date'])['opp_head'].count().unstack().add_prefix('sms_everyday_out_opp_head_count').reset_index().fillna(0)
sms_everyday_in_opp_head_count = sms[sms.in_out==1].groupby(['uid','date'])['opp_head'].count().unstack().add_prefix('sms_everyday_in_opp_head_count').reset_index().fillna(0)

out_opp_head_count = sms_everyday_out_opp_head_count.drop(['uid'],axis=1)
out_opp_head_count.columns=np.arange(1,46)
sms_every5day_out_opp_head_count = pd.DataFrame({'uid':sms_everyday_out_opp_head_count.uid})
sms_every5day_out_opp_head_count['1'] = out_opp_head_count.loc[:,1:5].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['2'] = out_opp_head_count.loc[:,6:10].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['3'] = out_opp_head_count.loc[:,11:15].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['4'] = out_opp_head_count.loc[:,16:20].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['5'] = out_opp_head_count.loc[:,21:25].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['6'] = out_opp_head_count.loc[:,26:30].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['7'] = out_opp_head_count.loc[:,31:35].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['8'] = out_opp_head_count.loc[:,36:40].apply(lambda x: x.sum(), axis=1)
sms_every5day_out_opp_head_count['9'] = out_opp_head_count.loc[:,41:45].apply(lambda x: x.sum(), axis=1)

in_opp_head_count = sms_everyday_in_opp_head_count.drop(['uid'],axis=1)
in_opp_head_count.columns=np.arange(1,46)
sms_every5day_in_opp_head_count = pd.DataFrame({'uid':sms_everyday_in_opp_head_count.uid})
sms_every5day_in_opp_head_count['1'] = in_opp_head_count.loc[:,1:5].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['2'] = in_opp_head_count.loc[:,6:10].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['3'] = in_opp_head_count.loc[:,11:15].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['4'] = in_opp_head_count.loc[:,16:20].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['5'] = in_opp_head_count.loc[:,21:25].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['6'] = in_opp_head_count.loc[:,26:30].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['7'] = in_opp_head_count.loc[:,31:35].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['8'] = in_opp_head_count.loc[:,36:40].apply(lambda x: x.sum(), axis=1)
sms_every5day_in_opp_head_count['9'] = in_opp_head_count.loc[:,41:45].apply(lambda x: x.sum(), axis=1)


# In[18]:


#sms_count_by_opp_num 
avg_count = sms_out_opp_num['sms_out_opp_numcount']/sms_out_opp_num['sms_out_opp_numunique_count']
sms_avg_out_count_by_opp_num = pd.DataFrame({'uid':sms_out_opp_num.uid,'sms_avg_out_count_by_opp_num':avg_count})
avg_count = sms_in_opp_num['sms_in_opp_numcount']/sms_in_opp_num['sms_in_opp_numunique_count']
sms_avg_in_count_by_opp_num = pd.DataFrame({'uid':sms_out_opp_num.uid,'sms_avg_in_count_by_opp_num':avg_count})


# In[19]:


sms_opp_head_count=sms.groupby(['uid','opp_head'])['uid'].count().unstack().add_prefix('sms_opp_head_').reset_index().fillna(0)


# ### 网站/APP记录

# In[20]:


# 不同wa数量/wa总数
w_a_name_cnt = wa.groupby(['uid','wa_type'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('w_name_').unstack().reset_index()

# 访问wa次数的各统计量
w_a_visit_cnt = wa.groupby(['uid','wa_type'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('w_a_visit_cnt').unstack().reset_index().fillna(0)

# 访问w/a时长的各统计量
w_a_visit_dura = wa.groupby(['uid','wa_type'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('w_a_visit_dura').unstack().reset_index().fillna(0)

## 每个用户w/a的上/下的流量
w_a_upflow = wa.groupby(['uid','wa_type'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('w_a_upflow_').unstack().reset_index().fillna(0)
w_a_downflow = wa.groupby(['uid','wa_type'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('w_a_downflow_').unstack().reset_index().fillna(0)

# 不同的日期数
wa_day_count = wa.groupby(['uid'])['date'].agg({'wa_day_count': lambda x: len(pd.unique(x))}).reset_index().fillna(0)

# 每天web上传流量
wa_everyday_web_up_flow = wa[wa.wa_type==0].groupby(['uid','date'])['up_flow'].sum().unstack().add_prefix('wa_everyday_web_up_flow').reset_index().fillna(0)

# 每天app上传流量
wa_everyday_app_up_flow = wa[wa.wa_type==1].groupby(['uid','date'])['up_flow'].sum().unstack().add_prefix('wa_everyday_app_up_flow').reset_index().fillna(0)

# 每天web下载流量
wa_everyday_web_down_flow = wa[wa.wa_type==0].groupby(['uid','date'])['down_flow'].sum().unstack().add_prefix('wa_everyday_web_down_flow').reset_index().fillna(0)

# 每天app下载流量
wa_everyday_app_down_flow = wa[wa.wa_type==1].groupby(['uid','date'])['down_flow'].sum().unstack().add_prefix('wa_everyday_app_down_flow').reset_index().fillna(0)

# 每天web访问时长
wa_everyday_web_visit_dura = wa[wa.wa_type==0].groupby(['uid','date'])['visit_dura'].sum().unstack().add_prefix('wa_everyday_web_visit_dura').reset_index().fillna(0)

# 每天app访问时长
wa_everyday_app_visit_dura = wa[wa.wa_type==1].groupby(['uid','date'])['visit_dura'].sum().unstack().add_prefix('wa_everyday_app_visit_dura').reset_index().fillna(0)


# In[21]:


# 上传流量最多的w名字
wa_w = wa[wa.wa_type==0]
wa_name = wa_w.groupby(['uid','wa_name'])['up_flow'].sum().unstack().add_prefix('up').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_up_webname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_up_webname['wa_most_up_webname'] = l


# In[22]:


# 上传流量最多的a名字
wa_w = wa[wa.wa_type==1]
wa_name = wa_w.groupby(['uid','wa_name'])['up_flow'].sum().unstack().add_prefix('up').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_up_appname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_up_appname['wa_most_up_appname'] = l


# In[23]:


# 下载流量最多的w名字
wa_w = wa[wa.wa_type==0]
wa_name = wa_w.groupby(['uid','wa_name'])['down_flow'].sum().unstack().add_prefix('down').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_down_webname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_down_webname['wa_most_down_webname'] = l


# In[24]:


# 下载流量最多的a名字
wa_w = wa[wa.wa_type==1]
wa_name = wa_w.groupby(['uid','wa_name'])['down_flow'].sum().unstack().add_prefix('down').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_down_appname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_down_appname['wa_most_down_appname'] = l


# In[25]:


# 访问次数最多的w名字
wa_w = wa[wa.wa_type==0]
wa_name = wa_w.groupby(['uid','wa_name'])['visit_cnt'].sum().unstack().add_prefix('cnt').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_cnt_webname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_cnt_webname['wa_most_cnt_webname'] = l


# In[26]:


# 访问次数最多的a名字
wa_w = wa[wa.wa_type==1]
wa_name = wa_w.groupby(['uid','wa_name'])['visit_cnt'].sum().unstack().add_prefix('cnt').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_cnt_appname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_cnt_appname['wa_most_cnt_appname'] = l


# In[27]:


# 访问时长最多的w名字
wa_w = wa[wa.wa_type==0]
wa_name = wa_w.groupby(['uid','wa_name'])['visit_dura'].sum().unstack().add_prefix('cnt').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_dura_webname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_dura_webname['wa_most_dura_webname'] = l


# In[28]:


# 访问时长最多的a名字
wa_w = wa[wa.wa_type==1]
wa_name = wa_w.groupby(['uid','wa_name'])['visit_dura'].sum().unstack().add_prefix('cnt').reset_index().fillna(0)
wa_uid = wa_name.uid;
wa_name = wa_name.drop(['uid'],axis=1)
col_name = np.arange(0,wa_name.shape[1])
col_name = col_name.astype('U')
wa_name.columns =col_name
wa_name_t = wa_name.T
col_name2 = np.arange(0,wa_name_t.shape[1])
col_name2 = col_name2.astype('U')
wa_name_t.columns =col_name2
ss = wa_name_t.idxmax()
wa_most_dura_appname = pd.DataFrame(wa_uid, columns=['uid'])
l = list(ss)
wa_most_dura_appname['wa_most_dura_appname'] = l


# In[29]:


# 前45种wa和其他剩余wa的dura
wa_w = wa[wa.wa_type==1]
wa_name_first_k_ori = wa_w['wa_name'].value_counts()[0:30]
wa_name_first_k = []
for i in np.arange(0,wa_name_first_k_ori.shape[0]):
    wa_name_first_k.append(wa_name_first_k_ori[i:i+1].to_string().split(' ')[0])

wa_name_else = wa_w[~wa_w.wa_name.isin(wa_name_first_k)]
wa_name_first_k_w_dura = wa_name_else.groupby(['uid'])['visit_dura'].sum().reset_index().fillna(0).rename(columns={'uid': 'uid', 'visit_dura': 'wa_name_else_dura'})
    
for num in wa_name_first_k:
    wa_name_k_dura = wa_w[wa_w.wa_name == num].groupby(['uid'])['visit_dura'].sum().reset_index().fillna(0)
    wa_name_first_k_w_dura = pd.merge(wa_name_first_k_w_dura,wa_name_k_dura,how='outer',on='uid')

wa_name_first_k_w_dura.fillna(0,inplace=True)
#wa_name_first_k_w_dura.head()


# In[30]:


# 前45种wa和其他剩余wa的dura
wa_w = wa[wa.wa_type==0]
wa_name_first_k_ori = wa_w['wa_name'].value_counts()[0:30]
wa_name_first_k = []
for i in np.arange(0,wa_name_first_k_ori.shape[0]):
    wa_name_first_k.append(wa_name_first_k_ori[i:i+1].to_string().split(' ')[0])

wa_name_else = wa_w[~wa_w.wa_name.isin(wa_name_first_k)]
wa_name_first_k_a_dura = wa_name_else.groupby(['uid'])['visit_dura'].sum().reset_index().fillna(0).rename(columns={'uid': 'uid', 'visit_dura': 'wa_name_else_dura'})
    
for num in wa_name_first_k:
    wa_name_k_dura = wa_w[wa_w.wa_name == num].groupby(['uid'])['visit_dura'].sum().reset_index().fillna(0)
    wa_name_first_k_a_dura = pd.merge(wa_name_first_k_a_dura,wa_name_k_dura,how='outer',on='uid')

wa_name_first_k_a_dura.fillna(0,inplace=True)
#wa_name_first_k_a_dura.head()


# In[31]:


# 将各个特征拼在一起
feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_dura,
           voice_opp_len_inout,voice_dura_type,voice_day_count,voice_everyday_out_count,voice_everyday_in_count,
           voice_hour_count,voice_everyday_in_dura,voice_everyday_out_dura,voice_everyday_call_type,
           voice_dura_cls_type,voice_dura_cls,voice_everyday_dura_type,
           #6.8
           voice_num_first_k_dura,voice_time,
           voice_opp_head_count,
           #6.9
           
           sms_out_opp_num,sms_in_opp_num,sms_opp_head,sms_out_opp_len,sms_in_opp_len,sms_in_out,
           sms_day_count,sms_everyday_out_count,sms_everyday_in_count,sms_hour_out_count,sms_hour_in_count,
            #6.7 add
           sms_every5day_out_opp_head_unique,sms_every5day_in_opp_head_unique,
           sms_every5day_out_opp_head_count,sms_every5day_in_opp_head_count,
           sms_avg_out_count_by_opp_num,sms_avg_in_count_by_opp_num,
           sms_opp_head_count,
    
           w_a_name_cnt,w_a_visit_cnt,w_a_visit_dura,w_a_upflow,w_a_downflow,
           wa_day_count,wa_everyday_web_up_flow,
           wa_everyday_app_up_flow,wa_everyday_web_down_flow,wa_everyday_app_down_flow,
           wa_everyday_web_visit_dura,wa_everyday_app_visit_dura,wa_most_up_webname,wa_most_down_webname,
           wa_most_cnt_webname,wa_most_dura_webname,wa_most_up_appname,wa_most_down_appname,wa_most_cnt_appname,
           wa_most_dura_appname,
           # 6.7add
           wa_name_first_k_w_dura,wa_name_first_k_a_dura,
           
            
          ]


# In[32]:


# 将train特征和test特征分离
train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')


# In[33]:


test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')


# In[34]:


# 填补Nan
train_feature.fillna(0,inplace=True)
test_feature.fillna(0,inplace=True)


# In[35]:


# 保存特征值
train_feature.to_csv('../data/train_featureV1.csv',index=None)
test_feature.to_csv('../data/test_featureV1.csv',index=None)


# In[36]:


train_feature.head()

