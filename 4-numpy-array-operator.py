#!/usr/bin/env python
# coding: utf-8

# # 1. Array vs Array Operation
# * Arrays can calculator with each other
#     * Applied element-wise when performing arithmetic operation.
#     * Arrays of the same sizes are required for element-wise operation.

# In[1]:


import numpy as np


# In[13]:


x = np.array([1,2,3])
y = np.array([4,5,6])
x,y


# In[14]:


x.shape


# In[4]:


y.shape


# In[7]:


x+yn # element-wise addition


# In[8]:


x-y # element-wise subtraction


# In[9]:


x*y # element-wise multiplication


# In[10]:


x/y # element-wise division


# In[16]:


x**y # element-wise power


# In[21]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
x, y


# In[22]:


x*y


# In[23]:


np.matmul(x,y) # matrix multiplication


# In[24]:


x = np.array([[1,2,3],[4,5,6]])
y = np.array([[1,2,3],[4,5,6]])
x,y


# In[25]:


x.shape


# In[26]:


y.shape


# In[27]:


x*y # elemen-wise multiplication


# In[30]:


x


# In[31]:


x.transpose()


# In[29]:


np.matmul(x.transpose(),y) # matrix multiplication 


# # 2. Array vs Python Scalar Operation
# * When calculating Array and scalar, calculate for each element of the array

# In[34]:


x = np.array([1,2,3])
x


# In[35]:


x/2 # each element divived by scalar


# In[36]:


x*5 # each element multiplied by scalar


# # 3. Broadcasting of Array
# * Calcualate by automatically adjusting the size of arrays of different sizes
#     * If the size of the dimension is 1, duplicate to the same size and calculate.

# In[38]:


x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y = np.array([1,2,3,4])
x, y 


# In[39]:


x.shape, y.shape


# In[44]:


y.reshape(-1,1)


# In[46]:


x + y.reshape(-1,1) # y.reshape(-1,1) is broadcasted from (4,1) to (4,4)


# In[50]:


x[..., np.newaxis]


# In[52]:


y[..., np.newaxis, np.newaxis]


# In[49]:


(x[..., np.newaxis] + y[..., np.newaxis, np.newaxis]).shape # matching shape before addition


# In[53]:


x[..., np.newaxis] + y[..., np.newaxis, np.newaxis]


# In[ ]:




