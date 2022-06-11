#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import tldextract
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr,spearmanr


# In[ ]:


leftdf=pd.read_csv('../data/MBFC/left.csv')
lcdf=pd.read_csv('../data/MBFC/Left-Center.csv')
centdf=pd.read_csv('../data/MBFC/Least%20Biased.csv')
rcdf=pd.read_csv('/data/MBFC/Right-Center.csv')
rightdf=pd.read_csv('../data/MBFC/right.csv')
scidf=pd.read_csv('../data/MBFC/pro-science.csv')
antiscidf=pd.read_csv('../data/MBFC/conspiracy-pseudoscience.csv')
qdf=pd.read_csv('../data/MBFC/Questionable%20Sources.csv')


# In[ ]:


leftdf=leftdf[leftdf['domain'].notna()]
lcdf=lcdf[lcdf['domain'].notna()]
centdf=centdf[centdf['domain'].notna()]
rcdf=rcdf[rcdf['domain'].notna()]
rightdf=rightdf[rightdf['domain'].notna()]
scidf=scidf[scidf['domain'].notna()]
antiscidf=antiscidf[antiscidf['domain'].notna()]
qdf=qdf[qdf['domain'].notna()]


# In[ ]:


for i in tqdm(range(len(leftdf['domain']))):
    if 'https://' in leftdf['domain'].iloc[i]:
        uri = tldextract.extract(leftdf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(leftdf['domain'].iloc[i].replace('http://',''))
    leftdf['domain'].iloc[i]='.'.join(uri[1:3])

for i in tqdm(range(len(lcdf['domain']))):
    if 'https://' in lcdf['domain'].iloc[i]:
        uri = tldextract.extract(lcdf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(lcdf['domain'].iloc[i].replace('http://',''))
    lcdf['domain'].iloc[i]='.'.join(uri[1:3])
    
for i in tqdm(range(len(centdf['domain']))):
    if 'https://' in centdf['domain'].iloc[i]:
        uri = tldextract.extract(centdf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(centdf['domain'].iloc[i].replace('http://',''))
    centdf['domain'].iloc[i]='.'.join(uri[1:3])

for i in tqdm(range(len(rcdf['domain']))):
    if 'https://' in rcdf['domain'].iloc[i]:
        uri = tldextract.extract(rcdf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(rcdf['domain'].iloc[i].replace('http://',''))
    rcdf['domain'].iloc[i]='.'.join(uri[1:3])

for i in tqdm(range(len(rightdf['domain']))):
    if 'https://' in rightdf['domain'].iloc[i]:
        uri = tldextract.extract(rightdf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(rightdf['domain'].iloc[i].replace('http://',''))
    rightdf['domain'].iloc[i]='.'.join(uri[1:3])

for i in tqdm(range(len(scidf['domain']))):
    if 'https://' in scidf['domain'].iloc[i]:
        uri = tldextract.extract(scidf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(scidf['domain'].iloc[i].replace('http://',''))
    scidf['domain'].iloc[i]='.'.join(uri[1:3])
for i in tqdm(range(len(antiscidf['domain']))):
    if 'https://' in antiscidf['domain'].iloc[i]:
        uri = tldextract.extract(antiscidf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(antiscidf['domain'].iloc[i].replace('http://',''))
    antiscidf['domain'].iloc[i]='.'.join(uri[1:3])
for i in tqdm(range(len(qdf['domain']))):
    if 'https://' in qdf['domain'].iloc[i]:
        uri = tldextract.extract(qdf['domain'].iloc[i].replace('https://',''))
    else:
        uri = tldextract.extract(qdf['domain'].iloc[i].replace('http://',''))
    qdf['domain'].iloc[i]='.'.join(uri[1:3])


# In[ ]:


domsdf=leftdf.append([lcdf,centdf,rcdf,rightdf,scidf,antiscidf,qdf])
domsdf=domsdf[['domain','factual']]

#Remove factually ambiguous/irrelevant sources
domsdf=domsdf[domsdf.domain!='facebook.com']
domsdf=domsdf[domsdf.domain!='wordpress.com']
domsdf=domsdf[domsdf.domain!='blogspot.com']


# In[ ]:


poldf=leftdf.append([lcdf,centdf,rcdf,rightdf])
poldf=poldf[['domain','polarity']]

#Remove politically ambiguous/irrelevant sources
poldf=poldf[poldf.domain!='facebook.com']
domsdf=domsdf[domsdf.domain!='wordpress.com']
domsdf=domsdf[domsdf.domain!='blogspot.com']


# In[ ]:


dic={'Very Low':0,'Low':0.2,'Mixed':0.4,'Mostly Factual':0.6,'High':0.8,'Very High':1}
domsdf['misinf_score']=domsdf['factual'].map(dic)


# In[ ]:


poldic={'Left':0,'Left-Center':0.25,'Least Biased':0.5,'Right-Center':0.75,'Right':1}
poldf['pol_score']=poldf['polarity'].map(poldic)


# In[ ]:


for i in range(len(poldf)):
    poldf['domain'].iloc[i]=poldf['domain'].iloc[i].strip()


# In[ ]:


domsdf.to_csv('../data/domsdf.csv',index=False)
poldf.to_csv('../data/poldf.csv',index=False)


# ### SCORE POLITICAL SHARING AND EXPOSURE

# In[ ]:


def foo(url):
    uri = tldextract.extract(url)
    uri = '.'.join(uri[1:3])
    return uri

def clean(links):
    links = [foo(x) for x in links]
    links = [x for x in links if x in valid_domains]
    return links


# In[ ]:


pol_res=pickle.load(open('../data/final_all_exposure.pkl','rb'))


# In[ ]:


valid_domains=poldf['domain'].tolist()
valid_domains=list(set(valid_domains))

#Uncomment "pol_res['rt_links']" to get results shown in Supplementary section
pol_res['links_gen']=pol_res['links']#+pol_res['rt_links']

tqdm.pandas()
pol_res['cleaned_links']=pol_res['links_gen'].progress_apply(clean)
pol_res['cleaned_full_exp']=pol_res['full_exp'].progress_apply(clean)

pol_res['len']=pol_res['cleaned_links'].apply(len)
pol_res['exp_len']=pol_res['cleaned_full_exp'].apply(len)

pol_res=pol_res[(pol_res['len']>2)&(pol_res['exp_len']>2)]

poldict=dict(poldf[['domain','pol_score']].values)

def mapper(li):
    return np.mean([poldict[x] for x in li if x in poldict])        

pol_res['political_gen']=pol_res['cleaned_links'].progress_apply(mapper)
pol_res['political_exp']=pol_res['cleaned_full_exp'].progress_apply(mapper)

pol_res.corr()

pol_res.to_pickle('../data/political_res_decoupled.pkl')


# ### SCORE MISINFORMATION SHARING AND EXPOSURE

# In[ ]:


misinf_res=pickle.load(open('/data/Coronavirus-Tweets/final_all_exposure.pkl','rb'))

def foo(url):
    uri = tldextract.extract(url)
    uri = '.'.join(uri[1:3])
    return uri

def clean(links):
    links = [foo(x) for x in links]
    links = [x for x in links if x in valid_domains]
    return links


# In[ ]:


misinf_dict=dict(domsdf[['domain','misinf_score']].values)
valid_domains=list(set(domsdf['domain'].tolist()))

misinf_res['links_gen']=misinf_res['links']#+misinf_res['rt_links']

tqdm.pandas()
misinf_res['cleaned_links']=misinf_res['links_gen'].progress_apply(clean)
misinf_res['cleaned_full_exp']=misinf_res['full_exp'].progress_apply(clean)

misinf_res['len']=misinf_res['cleaned_links'].apply(len)
misinf_res['rt_len']=misinf_res['cleaned_full_exp'].apply(len)

misinf_res=misinf_res[(misinf_res['len']>2)&(misinf_res['rt_len']>2)]

def mapper(li):
    return np.mean([misinf_dict[x] for x in li])

misinf_res['fact_gen']=misinf_res['cleaned_links'].progress_apply(mapper)
misinf_res['fact_exp']=misinf_res['cleaned_full_exp'].progress_apply(mapper)

misinf_res

misinf_res.to_pickle('../data/misinf_res_decoupled.pkl')

