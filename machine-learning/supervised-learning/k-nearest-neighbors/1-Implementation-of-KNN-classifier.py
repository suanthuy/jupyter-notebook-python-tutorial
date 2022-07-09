#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# plot_decision_boundary function will draw decision boundary of classification model.
# It only accept 2-d inputs with sparse labels rather than one-hot encoding
# Reference: https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html

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


dataset = load_iris() # load iris data from sklearn


# In[4]:


data = dataset["data"]


# In[5]:


data


# In[6]:


X = data[:,:2]         # get first two features for input
y = dataset["target"]  # get target for output labels


# In[7]:


model = KNeighborsClassifier(n_neighbors=1) # create KNN model with k=1
model.fit(X,y)                              # train model

plt.figure(figsize=(6,5))
plot_decision_boundary(model, X, y)
plt.title("KNN classifier with k=1")
plt.show()


# In[8]:


model = KNeighborsClassifier(n_neighbors=5) # create KNN model with k=5
model.fit(X,y)                              # train model

plt.figure(figsize=(6,5))
plot_decision_boundary(model, X, y)
plt.title("KNN classifier with k=5")
plt.show()


# In[9]:


model = KNeighborsClassifier(n_neighbors=7) # create KNN model with k=7
model.fit(X,y)                              # train model

plt.figure(figsize=(6,5))
plot_decision_boundary(model, X, y)
plt.title("KNN classifier with k=7")
plt.show()


# In[10]:


model = KNeighborsClassifier(n_neighbors=11) # create KNN model with k=11
model.fit(X,y)                              # train model

plt.figure(figsize=(6,5))
plot_decision_boundary(model, X, y)
plt.title("KNN classifier with k=11")
plt.show()


# In[11]:


W = np.array([[1.0, 0.0],
              [0.0, 0.1]])
X_tr = np.matmul(X,W)
X, X_tr


# In[12]:


model = KNeighborsClassifier(n_neighbors=7, p=1) # create KNN model with q=1
model.fit(X_tr,y)                                # train model
 
plt.figure(figsize=(6,5))
plot_decision_boundary(model, X_tr, y)
plt.title("KNN classifier with p=1")
plt.show()


# In[13]:


model = KNeighborsClassifier(n_neighbors=7, p=2) # create KNN model with q=2
model.fit(X_tr,y)                                # train model
 
plt.figure(figsize=(6,5))
plot_decision_boundary(model, X_tr, y)
plt.title("KNN classifier with p=2")
plt.show()


# In[14]:


def mahalanobis_dist(x0, x1, InvSigma):
    return np.sqrt((x0-x1) @ InvSigma @ np.transpose(x0-x1))


# In[15]:


plt.figure(figsize=(6,5))
model = KNeighborsClassifier(n_neighbors=7, metric=mahalanobis_dist, 
                             metric_params={"InvSigma": np.linalg.inv(np.cov(np.transpose(X_tr)))})
model.fit(X_tr, y)
plot_decision_boundary(model, X_tr, y)
plt.title("KNN Classifier with Mahalanobis Distance")
plt.show()


# In[ ]:




