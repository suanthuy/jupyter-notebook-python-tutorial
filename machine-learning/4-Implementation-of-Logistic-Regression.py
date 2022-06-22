#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# In[2]:


def plot_decision_boundary(model, x, y):
    # Reference: https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-
    x_min, x_min = X[:, 0].min() -.5, X[:,0].max() +.5
    y_min, y_max = X[:, 0].min() -.5, X[:,0].max() +.5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h))


# In[ ]:




