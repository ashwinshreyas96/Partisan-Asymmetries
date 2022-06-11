#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import ast
import pickle
import tldextract
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt


# In[ ]:


#Load rehydrated tweets from Coronavirus-Tweets directory
filespath='../data/Coronavirus-Tweets'
files=os.listdir(filespath)


# In[ ]:


def extractor(li):
    li=[x[0]['expanded_url']for x in li if len(x)>0]
    return li


# In[ ]:


exposed_df=pd.DataFrame(columns=['screen_name','rt_urls_list'])
generated_df=pd.DataFrame(columns=['screen_name','urls_list'])
full_expo_df=pd.DataFrame(columns=['screen_name','expo_urls_list'])


for file in tqdm(files):
    df=pickle.load(open(os.path.join(filespath,file),'rb'))
    generated=df[df['tweet_type']=='original']
    exposed=df[df['tweet_type']=='retweeted_tweet_without_comment']
    generated=generated[['screen_name','urls_list']]
    exposed=exposed[['screen_name','rt_urls_list']]
    
    generated=generated[generated['urls_list'].notna()]
    exposed=exposed[exposed['rt_urls_list'].notna()]
    
    exp_users=list(exposed['screen_name'].unique())
    
    tmp_df1=df[df['rt_tweetid'].notna()]
    tmp_df2=df[df.screen_name.isin(tmp_df1['rt_screen'].tolist())]
    tmp_df2=tmp_df2[['screen_name','urls_list']]
    tmp_df2['urls_list']=tmp_df2['urls_list'].apply(ast.literal_eval)
    tmp_df2=tmp_df2.groupby('screen_name')['urls_list'].apply(list).reset_index(name='expo_urls_list')
    tmp_df2=tmp_df2.rename(columns={'screen_name':'rt_screen'})
    sub_res=tmp_df2.merge(tmp_df1,on='rt_screen')
    sub_res=sub_res[['screen_name','expo_urls_list']]
    
    #generated['urls_list']=generated['urls_list'].apply(ast.literal_eval)
    #exposed['rt_urls_list']=exposed['rt_urls_list'].apply(ast.literal_eval)
    
    exposed_df=exposed_df.append(exposed,ignore_index=True)
    generated_df=generated_df.append(generated,ignore_index=True)
    full_expo_df=full_expo_df.append(sub_res,ignore_index=True)


# In[ ]:


tqdm.pandas()
exposed_df['rt_urls_list']=exposed_df['rt_urls_list'].progress_apply(ast.literal_eval)


# In[ ]:


generated_df['urls_list']=generated_df['urls_list'].progress_apply(ast.literal_eval)


# In[ ]:


full_expo_df['cleaned_expo_urls_list']=full_expo_df['expo_urls_list'].progress_apply(extractor)


# In[ ]:


exposed_df['len']=exposed_df['rt_urls_list'].apply(len)
generated_df['len']=generated_df['urls_list'].apply(len)
full_expo_df['len']=full_expo_df['cleaned_expo_urls_list'].apply(len)


# In[ ]:


exposed_df=exposed_df[exposed_df['len']>0]
generated_df=generated_df[generated_df['len']>0]
full_expo_df=full_expo_df[full_expo_df['len']>0]


# In[ ]:


exposed_df['rt_links']=exposed_df['rt_urls_list'].progress_apply(lambda x: x[0]['expanded_url'])
generated_df['links']=generated_df['urls_list'].progress_apply(lambda x: x[0]['expanded_url'])


# In[ ]:


del exposed_df['len']
del generated_df['len']
del full_expo_df['len']
del exposed_df['rt_urls_list']
del generated_df['urls_list']
del full_expo_df['expo_urls_list']


# In[ ]:


merged_full_expo_df=full_expo_df.groupby('screen_name')['cleaned_expo_urls_list'].progress_apply(sum).reset_index(name='full_exp')
groups_exposed=exposed_df.groupby('screen_name')['rt_links'].progress_apply(list).reset_index(name='rt_links')
groups_generated=generated_df.groupby('screen_name')['links'].progress_apply(list).reset_index(name='links')


# In[ ]:


res=groups_exposed.merge(groups_generated,on='screen_name')
res=res.merge(merged_full_expo_df,on='screen_name')


# In[ ]:


#Contains URLs exposed to and generated for each user
res.to_pickle('../data/final_all_exposure.pkl')

