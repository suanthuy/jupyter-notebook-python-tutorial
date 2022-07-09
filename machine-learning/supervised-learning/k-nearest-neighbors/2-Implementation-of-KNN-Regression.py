#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor


# In[3]:


dataset = load_boston()


# In[4]:


print(dataset.DESCR)


# In[5]:


data = dataset.data


# In[6]:


data.shape


# In[7]:


X = np.reshape(data[:, 12], (-1,1)) # use LSTAT feature
y = dataset.target


# In[20]:


model = KNeighborsRegressor(n_neighbors=11)   # create KNN regression model
model.fit(X, y)


# In[21]:


plt.figure(figsize=(6,5))
plt.scatter(X[:,0], y)    # scatter plot of training data

X_test = np.reshape(np.linspace(0.0, 40.0, 50),(-1,1)) # new input
pred_y = model.predict(X_test)  # predict for new inputs
plt.plot(X_test, pred_y, "r-")  # plot of test data
plt.show()


# In[ ]:




