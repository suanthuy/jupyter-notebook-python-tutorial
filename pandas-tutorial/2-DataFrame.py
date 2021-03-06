#!/usr/bin/env python
# coding: utf-8

# # DataFrame Class
# * Table-type class created by grouping multiple Series objects
#     * Generated by receiving tabular data and names of each column and row.
#     * Each column consists of Series objects.

# # 1. Creating DataFrame Objects

# In[2]:


import numpy as np
import pandas as pd

data = {"col1":[1,2,3,4], "col2":[5,6,7,8]}


# In[3]:


data


# In[5]:


df = pd.DataFrame(data) # DataFrame from dictionary

df


# In[6]:


data = np.random.rand(10,5)


# In[7]:


data


# In[14]:


indices = list(range(1,11))
indices


# In[9]:


columns = ["A" , "B","C","D","E"]


# In[10]:


df = pd.DataFrame(data) # DataFrame with only data


# In[11]:


df


# In[12]:


pd.DataFrame(data, columns=columns) # DataFrame with data and columns


# In[13]:


pd.DataFrame(data, columns=columns, index=indices) # DataFrame with data, columns and index


# In[19]:


sample1 = pd.Series(["Thomas", 80, "A"], index=["name", "math_score", "grade"], name=101)
sample2 = pd.Series(["Emily", 45, "C"], index=["name", "eng_score", "grade"], name=102)
sample1, sample2


# In[20]:


df = pd.DataFrame([sample1, sample2])


# In[21]:


df


# **=> the DataFrame is build with multiple Serires**

# In[ ]:




