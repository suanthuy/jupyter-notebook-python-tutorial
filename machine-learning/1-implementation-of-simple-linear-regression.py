#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sklearn')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression


# In[3]:


data_x, data_y = make_regression(n_samples=100, # number of samples
                                 n_features=1,  # number of features (dimension of data_x)
                                 noise=20.0,     # noise level (default: 0)
                                 random_state=101) # select random seed of reproducibility


# In[4]:


data_x.shape, data_y.shape


# In[5]:


plt.figure()
plt.scatter(data_x, data_y) # draw scatter plot of samples
plt.show()


# In[6]:


lr = LinearRegression() # create linear regression model
lr.fit(data_x, data_y)  # fit the model to data samples


# In[7]:


lr.coef_     # coefficients(slope) of linear regression model


# In[8]:


lr.intercept_   # bias of linear regression model


# In[9]:


x_new = np.linspace(-3,3,100).reshape(-1,1)
y_hat = lr.predict(x_new)    # predict outputs from new inputs


# In[10]:


plt.figure()
plt.scatter(data_x, data_y) # draw scatter plot of samples
plt.plot(x_new, y_hat, "r-") # draw predictions by red line
plt.show()


# In[ ]:




