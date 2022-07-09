#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification


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


# In[5]:


# generate classification dataset
X, y = make_classification(n_classes=3,            # total 3 classes
                           n_features=2,           # feature dimension is 2
                           n_informative=2,        # informative dimension is 2
                           n_redundant=0,          # redundant dimension is 0
                           n_clusters_per_class=1, # one cluster in one class
                           random_state=101)       # random seed fixed


# In[6]:


svm = LinearSVC(C=1.0, 
                multi_class="ovr") # one vs .rest strategy (default)
svm.fit(X, y)
plot_decision_boundary(svm, X, y)
plt.title("Multi-class classification by OvR")
plt.show()


# In[9]:


# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
# --> add max_iter=10000
svm = LinearSVC(C=1.0, 
                multi_class="crammer_singer", max_iter=10000) # one vs .one strategy 


svm.fit(X, y)
plot_decision_boundary(svm, X, y)
plt.title("Multi-class classification by OvO")
plt.show()


# In[ ]:




