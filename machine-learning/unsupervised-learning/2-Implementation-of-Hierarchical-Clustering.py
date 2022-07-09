#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons


# In[3]:


X, y = make_moons(n_samples=200,    # total 200 samples
                 noise=0.15)       # noise ratio 0.15


# In[5]:


hc = AgglomerativeClustering(n_clusters=2,        # hierarchical clustering number of clusters is 2
                             linkage="single")    # linkage method is set to single

y_pred = hc.fit_predict(X)
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], color="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], color="blue")
plt.title("Hierarchical Clustering /w Single Linkage")
plt.show()


# In[7]:


hc = AgglomerativeClustering(n_clusters=2,        # hierarchical clustering number of clusters is 2
                             linkage="complete")    # linkage method is set to single

y_pred = hc.fit_predict(X)
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], color="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], color="blue")
plt.title("Hierarchical Clustering /w Complete Linkage")
plt.show()


# In[8]:


hc = AgglomerativeClustering(n_clusters=2,        # hierarchical clustering number of clusters is 2
                             linkage="complete")    # linkage method is set to complete

y_pred = hc.fit_predict(X)
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], color="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], color="blue")
plt.title("Hierarchical Clustering /w Complete Linkage")
plt.show()


# In[9]:


hc = AgglomerativeClustering(n_clusters=2,        # hierarchical clustering number of clusters is 2
                             linkage="average")    # linkage method is set to average

y_pred = hc.fit_predict(X)
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], color="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], color="blue")
plt.title("Hierarchical Clustering /w Average Linkage")
plt.show()


# In[10]:


hc = AgglomerativeClustering(n_clusters=2,        # hierarchical clustering number of clusters is 2
                             linkage="ward")      # linkage method is set to ward

y_pred = hc.fit_predict(X)
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], color="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], color="blue")
plt.title("Hierarchical Clustering /w Ward Linkage")
plt.show()


# In[ ]:




