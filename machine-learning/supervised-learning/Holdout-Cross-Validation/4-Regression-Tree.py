#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, plot_tree


# In[2]:


dataset = load_boston()


# In[3]:


print(dataset.DESCR)


# In[4]:


data = dataset.data


# In[5]:


data.shape


# In[9]:


X = np.reshape(data[:, 12], (-1, 1))
y = dataset.target
X.shape, y.shape


# In[7]:


model = DecisionTreeRegressor()
model.fit(X, y)


# In[11]:


plt.figure(figsize=(6,5))
plt.scatter(X[:,0], y)

X_test = np.reshape(np.linspace(0.0, 40.0, 50), (-1,1))
pred_y = model.predict(X_test)
plt.plot(X_test, pred_y, "r-")
plt.show()


# In[18]:


X = np.reshape(data[:,12], (-1,1))
y = dataset.target
model = DecisionTreeRegressor(max_depth=5, ccp_alpha=0.02)
model.fit(X,y)


# In[19]:


plt.figure(figsize=(6,5))
plt.scatter(X[:,0], y)

X_test = np.reshape(np.linspace(0.0, 40.0, 50), (-1,1))
pred_y = model.predict(X_test)
plt.plot(X_test, pred_y, 'r-')
plt.show()


# In[20]:


plt.figure(figsize=(20,12))
plot_tree(model)
plt.show()


# In[ ]:




