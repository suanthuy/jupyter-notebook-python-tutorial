#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification, make_moons


# In[3]:


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


# In[4]:


# generate classification dataset
X, y = make_classification(n_samples=100,   # total 100 samples
                           n_features=2,    # feature dimension is 2
                           n_informative=2, # informative dimension is 2
                           n_redundant=0,   # redundant dimension is 0
                           n_clusters_per_class=1, # one cluster in one class
                           class_sep=0.3,   # class separability (higher, the better)
                           n_classes=2,     # number of classes is 2
                           random_state=1)  # random seed fixed


# In[5]:


svm = LinearSVC(C=1.0)   # svm classifier model with C=1.0
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("SVM Classification w/ C=1.0")
plt.show()


# In[8]:


svm = LinearSVC(C=0.2)   # svm classifier model with C=1.0
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("SVM Classification w/ C=0.2")
plt.show()


# In[9]:


svm = LinearSVC(C=0.01)   # svm classifier model with C=1.0
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("SVM Classification w/ C=0.01")
plt.show()


# In[11]:


# generates interleaving half circles
X, y = make_moons(n_samples=100,    # total 100 samples
                 random_state=1)   # random seed fixed


# In[14]:


svm = LinearSVC(C=1.0)   # svm classifier model with C=1.0
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("SVM Classification w/ C=1.0")
plt.show()


# In[13]:


svm = LinearSVC(C=0.01)   # svm classifier model with C=1.0
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("SVM Classification w/ C=0.01")
plt.show()


# In[ ]:




