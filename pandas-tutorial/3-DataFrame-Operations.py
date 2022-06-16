#!/usr/bin/env python
# coding: utf-8

# # 1. Select Data

# In[1]:


import numpy as np
import pandas as pd

data = np.random.rand(10,5)
indices = list(range(101,111))
columns = ["A","B","C","D","E"]

df = pd.DataFrame(data=data, columns=columns, index=indices)
df


# In[2]:


df.head() # show top 5 records


# In[4]:


df.tail(3) # shows last 3 records


# # 2. Accessing elements in DataFrame in a variety of ways

# In[8]:


df.at[101, "A"] # get value from DataFrame by [index, column] # [index, columnn]


# In[9]:


df.at[101, "B"] = 10000 # set value from DataFrame by [index, column]


# In[10]:


df


# In[11]:


df.iat[0,0] # get Value from DataFrame by 0-based indices


# In[12]:


df.iat[7,1] = 100 # set value in DataFrame by 0-based indices


# In[13]:


df


# In[14]:


df["B"] # get a column as Series from DataFrame by [column]


# In[16]:


df["B"][109] # get a value from Series by [index] # [column, index]


# In[54]:


# df[109] # Error


# In[17]:


df[["B","C"]]


# In[18]:


df["B"].at[109]


# In[19]:


df["B"].iat[8] # get a value frm Series by [0-based index]


# In[20]:


df.head()


# In[21]:


df.loc[101] # get a Series from DataFrame by .loc[index]


# In[22]:


df.loc[[101,102,103]] # get DataFrame from DataFrame by .loc[[indices]]


# In[28]:


df.loc[101]["B"] # get value from Series by [index]


# In[29]:


df.loc[101].at["B"] # get value from a Series by .at[index]


# In[30]:


df.loc[101].iat[1] # get value from Series by .iat[0-based index]


# In[31]:


df.iloc[0].iat[1]


# # 3. Select data by condition

# In[32]:


import numpy as np
import pandas as pd

data = np.random.rand(10,5)
indices = list(range(101,111))
columns = ["A","B","C","D","E"]
df = pd.DataFrame(data=data, columns=columns, index=indices)
df


# ## 3.1 Get DataFrame using single condition

# In[33]:


df > 0.5 # relation operation gives logical DataFrame


# In[35]:


df[df > 0.5] # select values using logical DataFrame


# In[36]:


df["A"] > 0.5 # relation operation gives logical Series


# In[38]:


df[df["A"] > 0.5] # select record by logical Series


# In[39]:


df.loc[105] > 0.5 # relation operation gives logical Series


# In[40]:


df.loc[df.loc[105] > 0.5] # loc cannot select by logical Series


# In[42]:


df


# ## 3.2 Get DataFrame using multiple condition

# In[43]:


df["B"] < 0.5


# In[44]:


df["C"] >= 0.2


# In[46]:


(df["B"] < 0.5) & (df["C"] >= 0.2)


# In[47]:


df[(df["B"] < 0.5) & (df["C"] >= 0.2)] # & for AND operation


# In[48]:


df[(df["B"] < 0.5) | (df["C"] >= 0.2)] # | for OR operation


# In[51]:


df[(df["B"] < 0.5) ^ (df["C"] >= 0.2)] # ^ for XOR operation


# In[52]:


df[~(df["B"] < 0.5) ] # ~ for AND operation


# In[60]:


type(df["B"] < 0.5)


# In[ ]:




