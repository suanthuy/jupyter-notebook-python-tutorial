#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles


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


X, y = make_moons(n_samples=100,
                  random_state=1)


# In[4]:


svm = SVC(kernel="linear",    # sum with linear kernel
          C=1.0)
svm.fit(X, y)
plot_decision_boundary(svm, X, y)
plt.title("Linear SVM w/ C=1.0")


# In[12]:


svm = SVC(kernel="poly",     # svm with polynomial kernel
          degree=5,          # maxium degree of polynomial=5
          C=0.25, gamma=1.0)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("Polynomial Kernel SVM w/ C=0.25")


# In[20]:


svm = SVC(kernel="rbf",     # svm with rbf kernel
          C=1.0)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("RBF Kernel SVM w/ C=1.0")


# In[19]:


svm = SVC(kernel="sigmoid",     # svm with rbf kernel
          C=1.0)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("Sigmoid Kernel SVM w/ C=1.0")


# In[21]:


# generate concentric circles
X, y = make_circles(n_samples=100,    # total 100 samples
                    random_state=1)   # random seed fixed


# In[25]:


svm = SVC(kernel="linear",     # svm with polynomial kernel
          degree=5,            # maxium degree of polynomial=5
          C=0.25)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("Linear Kernel SVM w/ C=0.25")


# In[26]:


svm = SVC(kernel="poly",     # svm with polynomial kernel
          degree=5,            # maxium degree of polynomial=5
          C=0.25)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("Polynomial Kernel SVM w/ C=0.25")


# In[27]:


svm = SVC(kernel="rbf",     # svm with rbf kernel
          C=0.25)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("RBF Kernel SVM w/ C=1.0")


# In[28]:


svm = SVC(kernel="sigmoid",     # svm with sigmoid kernel
          C=1.0)
svm.fit(X,y)
plot_decision_boundary(svm, X, y)
plt.title("Sigmoid Kernel SVM w/ C=1.0")


# In[ ]:




