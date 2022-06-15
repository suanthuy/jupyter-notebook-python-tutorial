#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Numpy Package

# In[1]:


import numpy as np


# # 2. Numpy Array and Indexing

# In[2]:


x = np.arange(10)


# In[3]:


x


# In[4]:


x[3] # indexing of numpy array


# In[ ]:





# In[5]:


type(x[3]) # numpy datatype


# In[6]:


type(3) # numpy datetpye id defferent from python datatype


# In[7]:


y = x[3] # indexed value is immutable


# In[8]:


y 


# In[9]:


y = 100 # it does not affect original array


# In[10]:


x


# # 3. NumPy Array and Slicing

# In[11]:


x


# In[12]:


y = x[3:6]


# In[13]:


y


# In[14]:


y[0] = 100 # Slice refers to the same data, and it affects the original array


# In[15]:


x


# In[16]:


x


# In[17]:


x[0:6]


# In[18]:


x[:6]


# In[19]:


x[4:]


# In[20]:


x[1:7:2]


# In[21]:


x


# In[22]:


x[::2]


# In[23]:


x[5:1:-1]


# In[24]:


y = x[::-1] # array revering


# In[25]:


y


# In[26]:


y[0] = 20


# In[27]:


y


# In[28]:


x


# # 4. Modifying values using slicing

# In[29]:


x = np.arange(10)


# In[30]:


x


# In[31]:


x[:3]


# In[32]:


x[:3] = 100 # assinging a value to slide -> automatically broadcast


# In[33]:


x


# In[34]:


x[2:4] = [5000, 40000] # slice can be replaced with the same size list


# In[35]:


x


# In[36]:


x[::2] = [9,9,9,9,9] # stepped slice can be replaced with the same size list 


# In[37]:


x


# In[38]:


np.arange(25)


# In[39]:


np.arange(25).reshape((5,-1)) # reshape (5, -1) automatically calculates 5*x = 25 -> x=5


# In[58]:


x = np.arange(24).reshape((6,-1))


# In[57]:


x


# # 5. Indexing and slicing of N-D Arrays

# In[42]:


x[2, 2]


# In[45]:


x[2][2] # first indexing gives a row vector from a matrix
        # second indexing gives a value from a row vector


# In[47]:


x[:, 1:3] # : select all value in that dimension # 1:3 -> from 1 to 3 and not include 3


# In[49]:


x[1:3, 1:3]


# In[50]:


x[1:3]


# In[61]:


x = np.arange(24).reshape((2, 3, 4)) # 2*3*4 = 24 elements can be reshaped


# In[62]:


x


# In[63]:


x.shape


# In[67]:


x[:,:,2] # all the rows and all the columns with index is 2


# In[68]:


x[..., 2] # ... will select all expect for the last dimension # = x[:, :, 2]


# In[69]:


x[..., 1,2] # ... will select all expect for the last two dimension # x[:, 1, 2]


# # 6. Add dummy dimension (dimensional with a length of 1)

# In[70]:


x.shape


# In[71]:


x.reshape((2,1,3,4)) # dimension with length=1 can be added (method1)


# In[72]:


x.reshape((1,2,3,4))


# In[73]:


x[:, np.newaxis, :, :] # dimension with length=1 can be added (method2)


# In[74]:


x[:, np.newaxis, :, :].shape


# In[75]:


x[..., np.newaxis].shape


# In[77]:


x[..., np.newaxis]


# In[ ]:




