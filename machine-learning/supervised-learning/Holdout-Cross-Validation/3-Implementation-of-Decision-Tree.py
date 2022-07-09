#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[2]:


def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:,1].max() + .5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, shading="nearest", cmap=plt.cm.Paired)
    
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())


# In[3]:


iris = load_iris(as_frame=True)  # load Iris data ("data" field as DataFrame)


# In[4]:


df = iris["data"]  # get "data" field (DataFrame)


# In[5]:


df.columns


# In[6]:


X = df[["sepal length (cm)", "sepal width (cm)"]]   # get specific columns for input feature
y = iris["target"]                                  # get target output


# In[7]:


tree = DecisionTreeClassifier(max_depth=5)   # decision tree depth=5, w/o prunings
tree.fit(X, y)


# In[8]:


X_val = X.values
y_val = y.values


# In[9]:


plot_decision_boundary(tree, X_val, y_val)
plt.show()


# In[12]:


iris.keys()


# In[13]:


fig = plt.figure(figsize=(25,30))
_ = plot_tree(tree, feature_names=["sepal length", "sepal width"],
                   class_names=iris.target_names,
                   filled=True)
plt.show()


# In[14]:


tree = DecisionTreeClassifier(max_depth=5, ccp_alpha=0.01) # decision tree depth=5, w/pruning
tree.fit(X,y)


# In[15]:


X_val = X.values
y_val = y.values
plot_decision_boundary(tree, X_val, y_val)
plt.show()


# In[ ]:




