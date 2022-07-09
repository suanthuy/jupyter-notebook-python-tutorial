#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons


# In[2]:


X, y = make_moons()            # generate interleaving circles


# In[3]:


plt.scatter(X[:,0], X[:,1],    # scatter data samples
            c=y)               # coloring by label
plt.title("Moons dataset")
plt.show()


# In[4]:


pca = PCA(n_components=2)       # create PCA with result dimension 2
pca.fit(X)                      # fit model to data
X_pca = pca.transform(X)        # apply dimensionlity reduction

plt.scatter(X_pca[:,0], X_pca[:,1],    # scatter data samples
            c=y)                       # coloring by label
plt.title("Moons dataset with PCA")
plt.show()


# In[5]:


kpca = KernelPCA(kernel="rbf",        # create KPCA with rbf
                 gamma=20)            # gamma set to 20
kpca.fit(X, y)                        # fit model to data
X_kpca = kpca.transform(X)            # apply dimensionality reduction

plt.scatter(X_kpca[:,0], X_kpca[:,1], # scatter data samples
            c=y)                      # coloring by label
plt.title("Moons dataset with KPCA")
plt.show()


# In[ ]:




