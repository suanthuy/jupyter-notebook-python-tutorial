#!/usr/bin/env python
# coding: utf-8

# # 1. Set DataFrame Index

# In[1]:


import numpy as np
import pandas as pd

data = np.random.rand(10,5)
indices = list(range(101,111))
columns = ["A","B","C","D","E"]
df = pd.DataFrame(data=data, columns=columns, index=indices)
df


# In[2]:


df.reset_index().set_index("A") # change index and keep the original


# In[3]:


df


# In[4]:


df.set_index("A") # change index and keep original


# In[5]:


df


# In[ ]:


df.set_index("B", inplace=True) # index = B
df


# In[7]:





# In[8]:


df.set_index("D", inplace=True) # index = D (B is removed)
df


# # 2. Multi-Index of DataFrame

# In[24]:


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


# In[25]:


df.set_index(["class", "student_id"], inplace=True) # select multi-index


# In[26]:


df


# In[22]:


# data = {
#     "class": ["A","A","A","B","B","B","C","C"],
#     "student_id":[1,2,3,4,5,6,7,8],
#     "math_score":[60,70,50,64,75,20,60,90],
#     "eng_score":[66,56,90,34,55,56,62,44]
# }
# df = pd.DataFrame(data=data)
# df


# In[23]:


# df.set_index(["class", "student_id"], inplace=True) # select multi-index
# df


# In[27]:


df.xs("A") # select index (level=0)


# In[28]:


df.xs(1, level=1) # select index (level 1)


# In[29]:


df.xs(("A",1)) # select nulti indicies


# In[33]:


df.xs("math_score", axis=1) # select column


# # 3. Data aggregation

# In[35]:


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


# In[36]:


df.groupby(by="class") # groupby gives grouped iterator


# In[72]:


for name, group in df.groupby(by="0", axis=0): # groupby iterators group's name and DataFrame
    print("name:", name)
    print("group:", group)


# In[38]:


for name, group in df.groupby(by="class"): # groupby iterators group's name and DataFrame
    print(name)
    print(group)


# In[40]:


df.groupby(by="class").mean() # aggregated using mean() after grouping (group as index)


# In[41]:


df.groupby(by="class", as_index=False).mean() # aggregated using mean() after grouping


# In[42]:


df.set_index(["class","student_id"], inplace=True)
df


# In[44]:


df.groupby(level=0).sum() # groupby using multi-index level


# In[45]:


df.groupby(level=1).mean() # groupby using multi-index level


# In[46]:


df.groupby(level="student_id").mean() # groupby using multi-index level name


# In[47]:


df.groupby("class").aggregate(np.sum) # aggregate by passing function


# In[50]:


df.groupby("class").aggregate(lambda x: sum(map(lambda a: a**2, x))) # aggregate by passing lambda function


# In[ ]:




