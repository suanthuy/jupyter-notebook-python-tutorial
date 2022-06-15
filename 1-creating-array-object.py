#!/usr/bin/env python
# coding: utf-8

# # 1. Update Pip Version

# In[1]:


pip install --upgrade pip


# # 2. Installing Numpy Package

# In[2]:


pip install numpy


# # 3. Creating Array Object

# In[3]:


import numpy as np


# # 4. Create Numpy Array from data

# In[4]:


python_list = [1, 2, 3, 4]


# In[5]:


python_list


# In[6]:


x = np.asarray(python_list, dtype=np.float32) # Numpy array from python list (float)


# In[7]:


x


# In[8]:


x = np.asarray(python_list, dtype=np.uint32) # Numpy array from python list (int)


# In[9]:


x


# In[10]:


x = np.asarray([[1, 2],[3, 4]], dtype=np.float32) # 2-D Numpy array


# In[11]:


x


# In[12]:


x = np.asfortranarray([[1, 2], [3, 4]], dtype=np.float32) # Implemented using Fortran order


# In[13]:


x


# In[14]:


x.flags # F_CONTIGUOUS is True (elements are aligned in Fortran way)


# # 5. Creating Numpy Array filled with specific values

# ## 5.1 Creating Numpy Array filled with zeros

# In[15]:


np.zeros(10) # All zero array with shape (N,)


# In[16]:


np.zeros((10, 10)) # All zero 2-D array with shape (N,M)


# In[17]:


np.zeros((5,5,5)) # All zero 3-D array with shape (N,M,K)


# In[18]:


np.zeros((5,5,5)).shape


# In[19]:


np.ones(10)


# In[20]:


np.ones((5,5))


# In[21]:


x


# In[22]:


x.shape


# ## 5.2 Creating Array with the same shape

# In[24]:


np.zeros(x.shape) # creating all-zero with the same shape (method1)


# In[25]:


np.zeros_like(x) # creating all-zero with the same shape (method2)


# In[28]:


np.ones_like(x) # creating all-one with the same shape 


# In[ ]:





# In[29]:


np.eye(5) # I matrix (identity matrix) with shape (N,N)


# In[31]:


np.identity(5) # I matrix (identity matrix) with shape (N,N)


# In[32]:


np.eye(10,5)


# In[33]:


np.eye(5,10)


# In[34]:


np.ones(5)*3 # get all-constant array with shape (N,)


# In[35]:


np.full((5,10),3,dtype=np.float32) # get all-constant array with shape (N, M)


# In[37]:


np.full_like(x, 5) # get all-constant array with the same shape


# # 6. Python range and Numpy Range, Linspace and Logspace

# In[39]:


list(range(10)) # 0-9 integers (python_list)


# In[40]:


list(range(1,10))


# In[44]:


type(range(1,10))


# In[41]:


list(range(1,10,2))


# In[45]:


np.arange(10) # 0~9 integer (numpy)


# In[43]:


np.arange(1,10) # 1~9 integers (numpy)


# In[46]:


np.arange(1,10,2) # (start, end , step)


# In[48]:


np.arange(1.2,2,0.01)


# In[49]:


np.linspace(0.0,10.0,100) # divide [0.0, 10.0] space linearly by 100 numbers


# In[50]:


np.linspace(0.0, 10.0, 101)


# In[51]:


np.logspace(0.0, 10.0, 100) # divide [0.0, 10.0] space logarithmically by 100 numbers


# # 7. Create NumPy Array filled with values drawn randomly

# In[52]:


np.random.rand(5)


# In[54]:


np.random.rand(5,5)


# In[56]:


np.random.randint(10)


# In[57]:


np.random.randint(-5,5)


# In[58]:


np.random.randn(5) # random number from normal distribution (Gaussian with mu=0, std=1) with shape (N,)


# In[59]:


np.random.randn(5,5)


# In[ ]:




