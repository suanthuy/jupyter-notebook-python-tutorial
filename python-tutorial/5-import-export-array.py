#!/usr/bin/env python
# coding: utf-8

# # 1. Import/Export Array
# ## Serialization using pickle package

# In[1]:


import pickle
import numpy as np


# In[2]:


x = np.random.randn(10)
y = np.random.randn(5)
x, y


# ## Save and load an Array using pickle

# In[3]:


with open("x.p", "wb") as f:
    pickle.dump(x, f)


# In[4]:


with open("x.p", "rb") as f:
    data = pickle.load(f)


# In[5]:


data


# ## Save and load multiple Arrays

# In[7]:


with open("xy.p", "wb") as f:
    pickle.dump({"x":x, "y":y}, f)


# In[8]:


with open("xy.p", "rb") as f: # load multiple array objects
    data = pickle.load(f)


# In[9]:


data["x"]


# In[10]:


data["y"]


# ## Save and load using NumPy methods

# ### Save and load an array

# In[12]:


np.save("x.npy", x) # save using numpy method


# In[13]:


data = np.load("x.npy") # load using numpy method


# In[14]:


data


# ### Save and load multiple Arrays

# In[15]:


np.savez("xy.npz",x,y) # save multi object using numpy method


# In[16]:


data = np.load("xy.npz")


# In[17]:


data.files # anonymous objects


# In[18]:


data["arr_0"]


# In[19]:


data["arr_1"]


# In[20]:


np.savez("xy.npz", x=x, y=y)


# In[21]:


data = np.load("xy.npz")


# In[22]:


data.files


# In[23]:


data["x"]


# In[24]:


data["y"]


# In[ ]:




