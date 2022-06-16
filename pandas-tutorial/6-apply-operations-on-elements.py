#!/usr/bin/env python
# coding: utf-8

# # 1. Apply Operations on Elements

# In[1]:


import numpy as np
import pandas as pd

data = np.random.rand(10,5)
indices = list(range(101,111))
columns = ["A","B","C","D","E"]
df = pd.DataFrame(data=data, columns=columns, index=indices)
df


# In[2]:


df["A_Doubled"] = df["A"].map(lambda x: x*2) # "map" on a column (element-wise)
df


# In[3]:


df["A"][101]


# In[4]:


df.loc[101]


# In[5]:


df.at[101,"A"]


# In[6]:


df.groupby(by="A")


# In[7]:


df["sum"] = df.apply(np.sum, axis=1) # apply on each row
df


# In[8]:


df.loc["sum"] = df.apply(np.sum, axis=0) # apply on each column
df


# In[9]:


df.applymap(lambda x: x**2) # apply on each value of DataFrame (element-wise)


# # 2. Find Unique Values

# In[11]:


import numpy as np
import pandas as pd

data = {
    "class": ["A","A","A","B","B","B","C","C","C"],
    "student_id":[1,2,3,1,2,3,1,2,3],
    "math_score":[60,70,50,64,75,20,60,90,45],
    "eng_score":[66,56,90,34,55,56,62,44,49]
}
df = pd.DataFrame(data=data)
df


# In[13]:


df["student_id"].unique() # find unique values in Series


# In[14]:


df["class"].value_counts() # count unique values in Series


# In[17]:


df["class"].unique()


# In[16]:


df.value_counts() # get value count in DataFrame


# In[ ]:




