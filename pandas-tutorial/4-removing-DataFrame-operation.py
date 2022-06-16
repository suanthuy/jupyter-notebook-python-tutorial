#!/usr/bin/env python
# coding: utf-8

# # 1. Removing data

# In[1]:


import numpy as np
import pandas as pd

data = np.random.rand(10,5)
indices = list(range(101,111))
columns = ["A","B","C","D","E"]
df = pd.DataFrame(data=data, columns=columns, index=indices)


# In[2]:


df


# In[3]:


df.drop("A", axis=1) # drop a column


# In[4]:


df.drop(index=105)


# In[5]:


df.drop("A", axis=1, inplace=True)


# In[6]:


df


# In[7]:


df.drop(["B","C"], axis=1) # drop multiple columns


# In[8]:


df.drop([104], axis=0) # drop index


# In[9]:


df.drop([101,102,103], axis=0) # drop multiple indices


# In[10]:


df[~(df["B"] > 0.5)].index


# In[11]:


df.drop(df[~(df["B"] > 0.5)].index, axis=0) # drop indicies by condition


# # 2. Removing missing values

# In[12]:


import numpy as np
import pandas as pd

data = np.random.rand(10,5)
data[4,2] = np.nan
data[4,3] = np.nan
data[2,4] = np.nan
data[7,3] = np.nan
indices = list(range(101,111))
columns = ["A","B","C","D","E"]

df = pd.DataFrame(data=data, columns=columns, index=indices)
data, df


# In[13]:


df.isna() # find NaN


# In[14]:


df.notna() # find none NaN


# In[15]:


df.fillna(-1) # fill na with a constant


# In[16]:


df.dropna(axis=0) # remove na by dropping indices


# In[17]:


df.dropna(axis=1) # remove na by dropping columns


# In[18]:


df.dropna(how="all", axis=0) # remove na only if all values are na


# In[19]:


df.dropna(how="any", axis=0) # remove na only if a value is na


# In[25]:


df.dropna(thresh=1, axis=0) # remove na only if more than thresh values are missing


# In[27]:


df.dropna(subset=["A","C"], axis=0) # drop by only checking some columns


# In[ ]:




