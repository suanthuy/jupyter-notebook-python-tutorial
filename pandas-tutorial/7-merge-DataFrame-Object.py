#!/usr/bin/env python
# coding: utf-8

# # 1. Merge DataFrame Object

# In[1]:


import numpy as np
import pandas as pd

data1 = {"student_id": [101,102,103,104,105,106],
         "math_score": [60,70,50,64,75,20]}
data2 = {"student_id": [107,108,109],
         "math_score": [66,72,63]}
data3 = {"student_id": [101,102,103,104,105,106],
         "eng_score": [34,55,56,62,44,49]}
data4 = {"student_id": [104,105,106,107,108,109],
         "eng_score": [34,55,56,62,44,49]}

df1 = pd.DataFrame(data=data1)
df2 = pd.DataFrame(data=data2)
df3 = pd.DataFrame(data=data3)
df4 = pd.DataFrame(data=data4)

df1.set_index("student_id", inplace=True)
df2.set_index("student_id", inplace=True)
df3.set_index("student_id", inplace=True)
df4.set_index("student_id", inplace=True)

print(df1, df2, df3 ,df4, sep="\n")


# # 2. Merge Incomplete DataFrames

# In[2]:


df1


# In[3]:


df2


# In[4]:


df = pd.concat([df1, df2], axis=0) # concatenate two DataFrame vertically


# In[5]:


df


# In[6]:


df = pd.concat([df1,df3], axis=0)
df


# In[7]:


df = pd.concat([df1,df3], join="inner", axis=0) # inner mode: join only if both have values (intersection)
df


# In[8]:


df1


# In[9]:


df3


# In[10]:


pd.merge(df1, df3, left_index=True, right_index=True) # merge two DataFrame by their indices(student_id)


# In[11]:


pd.merge(df1,df2)


# In[12]:


pd.merge(df1.reset_index(),df3.reset_index())


# In[13]:


pd.merge(df1.reset_index(), df3.reset_index(), left_on="student_id", right_on="student_id")


# In[14]:


df1.join(df3) # same as pd.merge(df1, df3)


# In[15]:


df1.join(df4)


# In[16]:


df1.join(df4, how="outer") # outer: union(keep it there's any)


# In[17]:


df1.join(df4, how="inner") # inner: intersection (keep if there're in both)


# In[18]:


df1.join(df4, how="left") # keep record depending on left (df1)


# In[19]:


df1.join(df4, how="right") # keep record depending on right (df4)


# # 3. Sorting DataFrame

# In[21]:


import numpy as np
import pandas as pd

data = {"class":["A","A","A","B","B","B","C","C","C"],
        "student_id":[1,2,3,1,2,3,1,2,3],
        "math_score":[60,70,50,64,75,20,60,90,45],
        "eng_score":[66,56,90,34,55,56,62,44,49]}
df=pd.DataFrame(data=data)
df


# In[22]:


df.sort_values(by="math_score") # sort by math_score (ascending)


# In[24]:


df.sort_values(by="math_score", ascending=False) # sort by math_score (descending)


# In[25]:


df.sort_values(by=["class","math_score"], ascending=False) # sort by multi-criteria (descending)


# In[26]:


df.sort_values(by=["class", "math_score"], ascending=[True, False]) # sort by multi-criteria (ascending,descending)


# # 4. Pivot Table

# In[28]:


import numpy as np
import pandas as pd

data = {"class":["A","A","A","B","B","B","C","C","C"],
        "student_id":[1,2,3,1,2,3,1,2,3],
        "math_score":[60,70,50,64,75,20,60,90,45],
        "eng_score":[66,56,90,34,55,56,62,44,49]}
df=pd.DataFrame(data=data)
df


# # 5. Rearrange values by pivot table

# In[29]:


df.pivot_table(columns="student_id", index="class") # pivot table to rearrange table


# In[30]:


df.pivot_table(columns="student_id", index="class", values=["eng_score", "math_score"])


# In[31]:


df.pivot(columns="student_id")


# # 6. Data aggregation using pivot table

# In[32]:


df.pivot_table(columns="student_id", aggfunc="mean") # pivot table with aggregation


# In[33]:


df.pivot_table(columns="student_id", aggfunc="sum") # pivot table with aggregation


# In[34]:


df.pivot_table(columns="student_id",values=["eng_score", "math_score"], aggfunc=lambda x: sum(map(lambda a:a**2, x))) # pivot table with custom aggregation


# In[36]:


df.pivot_table(columns="class", aggfunc="mean", values=["math_score", "eng_score"]) # pivot table with aggregation


# In[ ]:




