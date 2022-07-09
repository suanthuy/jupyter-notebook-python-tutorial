#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(101)          # fix random seed for reproducibility
X = np.random.randn(50, 2)   # draw random inputs standard normal distribution
y = X[:,0] * 0.5 + X[:,1] * 0.9 + 0.2 + np.random.randn(*X[:,0].shape) # create random outputs 


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% for train, 30% for test


# # Linear Regression

# In[7]:


lr = LinearRegression()      # create linear regression model
lr.fit(X_train, y_train)     # train model


# # Ridge Regression

# In[8]:


ridge = Ridge(alpha=1.0)    # create ridge regression model
ridge.fit(X_train, y_train) # train model


# # LASSO

# In[9]:


lasso = Lasso(alpha=1.0)    # create lasso regression model
lasso.fit(X_train, y_train) # train model


# # Elastic Net

# In[10]:


elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # create elastic-net model
elastic.fit(X_train, y_train)                  # train_model


# In[11]:


pred_lr = lr.predict(X_test)           # get prediction of linear regression
pred_ridge = ridge.predict(X_test)     # get prediction of ridge regression
pred_lasso = lasso.predict(X_test)     # get prediction of lasso regression
pred_elastic = elastic.predict(X_test) # get prediction of elastic-net regression


# In[13]:


print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("Ridge Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_ridge)))
print("LASSO RMSE:", np.sqrt(mean_squared_error(y_test, pred_lasso)))
print("Elastic Net RMSE:", np.sqrt(mean_squared_error(y_test, pred_elastic)))


# In[ ]:




