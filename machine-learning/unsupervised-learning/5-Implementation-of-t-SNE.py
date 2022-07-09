#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_circles, make_moons, fetch_openml


# In[2]:


markers = ["o", "x"]
colors = ["blue", "red"]


# In[4]:


X, y = make_moons(n_samples=1000,        # total 1000 samples
                  noise=0.1)            # noise ratio 0.1


# In[5]:


for idx, mc in enumerate(zip(markers, colors)):
    plt.scatter(X[y==idx, 0], X[y==idx, 1], marker=mc[0], color=mc[1], alpha=0.8)
plt.title("Moons dataset")
plt.show()


# In[6]:


pca = PCA(n_components=2)         # PCA dimensionality reduction to 2
pca.fit(X)                        # fit PCA on data
X_pca = pca.transform(X)          # apply dimensionality reduction

for idx, mc in enumerate(zip(markers, colors)):
    plt.scatter(X_pca[y==idx, 0], X_pca[y==idx, 1], marker=mc[0], color=mc[1], alpha=0.8)
plt.title("Moons dataset with PCA")
plt.show()


# In[ ]:


kpca = KernelPCA(kernel="rbf",    # KPCA dimensionality reduction to 2
                 gamma=15.0)         
pca.fit(X)                        # fit PCA on data
X_pca = pca.transform(X)          # apply dimensionality reduction

for idx, mc in enumerate(zip(markers, colors)):
    plt.scatter(X_pca[y==idx, 0], X_pca[y==idx, 1], marker=mc[0], color=mc[1], alpha=0.8)
plt.title("Moons dataset with PCA")
plt.show()

