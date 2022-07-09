#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_wine


# In[2]:


wine = load_wine()


# In[3]:


X = wine["data"]
y = wine["target"]


# In[4]:


X.shape


# In[5]:


pca = PCA(n_components=2)   # dimension reduced to 2
pca.fit(X)                  # fit PCA model on data
X_pca = pca.transform(X)    # apply dimensionaly reduction


# In[6]:


X_pca.shape


# In[7]:


plt.scatter(X_pca[:,0], X_pca[:,1],      # scatter plot
            c=y)                         # coloring by label
plt.title("Wine dataset with PCA")
plt.show()


# In[8]:


lda = LinearDiscriminantAnalysis(n_components=2)    # create LDA model
lda.fit(X,y)                                        # LDA fits to data and target (supervised)
X_lda = lda.transform(X)                            # apply LDA


# In[9]:


plt.scatter(X_lda[:,0], X_lda[:,1],                 # scatter plot
            c=y)                                    # coloring by label

plt.title("Wine dataset with LDA")
plt.show()


# In[ ]:




