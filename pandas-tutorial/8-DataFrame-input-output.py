#!/usr/bin/env python
# coding: utf-8

# # 1. CSV Files Input/Output

# In[1]:


import pandas as pd


# In[2]:


# download train.csv file from https://www.kaggle.com/c/titanic/data?select=train.csv
df = pd.read_csv("train.csv")
df.head()


# In[3]:


df


# In[4]:


df.set_index("PassengerId", inplace=True)


# In[5]:


df


# In[6]:


df.to_csv("trainv2.csv")


# In[7]:


df = pd.read_csv("trainv2.csv")
df.head()


# In[8]:


df.set_index("PassengerId", inplace=True)
df


# In[9]:


df.to_csv("trainv2.csv") # save DataFrame as csv


# In[10]:


df = pd.read_csv("trainv2.csv")
df


# In[11]:


df.to_csv("trainv3.csv") # not using set index will create Unnamed indices


# In[14]:


df = pd.read_csv("trainv3.csv")
df


# In[15]:


df.to_csv("trainv4.csv", index=False)


# In[17]:


df = pd.read_csv("trainv4.csv")
df


# # 2. Import Sqlite DB

# In[18]:


import sqlite3
import pandas as pd


# In[21]:


# download database.sqlite file from https://www.kaggle.com/datasets/kaggle/sf-salaries?select=database.sqlite
conn = sqlite3.connect("database.sqlite")


# In[24]:


pd.read_sql(con=conn, sql='SELECT name FROM sqlite_master WHERE type="table"') # read a table


# In[25]:


df = pd.read_sql(con=conn, sql="SELECT * FROM Salaries") # read all columns from a table


# In[26]:


df.head()


# # 3. Import Table from  Web Page

# In[53]:


get_ipython().system('pip install lxml')


# In[38]:


import pandas as pd


# In[54]:


# https://www.ssa.gov/international/coc-docs/states.html
df.list = pd.read_html("https://www.ssa.gov/international/coc-docs/states.html") # read all tables in html


# In[56]:


df = df.list[0] # Select first element in the list


# In[42]:


df


# In[33]:


df.head()


# In[44]:


df.list[0]


# In[ ]:




