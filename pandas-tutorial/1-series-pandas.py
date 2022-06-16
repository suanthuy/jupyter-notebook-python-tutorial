#!/usr/bin/env python
# coding: utf-8

# # 1. Series Class
# * A class that deals with multiple indexed values
#     * Data structure similar to Python's list or dictionary data type.
#     * Create an object by receiving multiple values and indexes of each value.
#     * If you don't enter an index, 0-based index is used.

# # 2. Creating Series Objects 

# In[3]:


import pandas as pd

data = [10,5,87,32]
indices = ["Thomas", "Emily", "Sam", "Scott"]


# In[4]:


x = pd.Series(data) # Series from only data


# In[5]:


x


# In[6]:


x.index


# In[7]:


x = pd.Series(data, indices) # Series from data, and indices


# In[8]:


x


# In[9]:


x.index


# In[12]:


x.values


# In[13]:


x["Thomas"]


# In[14]:


x = pd.Series(data, index=indices, name="math_Score")


# In[15]:


x


# In[16]:


data_dict = {k:v for k, v in zip(indices, data)}


# In[17]:


data_dict


# In[21]:


x = pd.Series(data_dict)


# In[22]:


x


# In[ ]:




