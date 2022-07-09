#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[3]:


np.random.seed(101)    # fix random seed for reproducibility
x = np.random.randn(20) # draw random input1 from standard normal distribution
y = np.random.randn(20) # draw random input2 from standard normal distribution

X, Y = np.meshgrid(x,y)  # create mesh grid
Z = 1.3*X + 0.1*Y + 0.42 + np.random.randn(*X.shape)   # create random output


# In[15]:


Z.shape


# In[14]:


X.shape, Y.shape


# In[16]:


np.reshape(X,-1)


# In[6]:


np.stack([np.reshape(X, -1), np.reshape(Y, -1)], axis=1).shape


# In[7]:


model = LinearRegression()   # create linear regression model
feat = np.stack([np.reshape(X,-1), np.reshape(Y,-1)], axis=1) # form feature vector stack ((400,2) Array)
label = np.reshape(Z, -1)    # form output ((400, ) Array)
model.fit(feat, label)       # train a linear regression label


# In[23]:


feat.shape


# In[25]:


fig = plt.figure()
ax = plt.axes(projection="3d")   # make a 3-d axe for surface plot
ax.scatter(X,Y,Z)                # plot sample data

# create 10x10 mesh grid covers the data sample
xx, yy = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
zz = np.zeros_like(xx)

# get model prediction from 10x10 mesh grid 
zz = np.reshape(model.predict(np.stack([np.reshape(xx[:], -1), np.reshape(yy, -1)], axis=1)), zz.shape)
ax.plot_surface(xx, yy, zz, color="r", alpha=0.5)   # draw surface plot of prediction

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")


# In[ ]:




