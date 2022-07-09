#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


# In[3]:


dataset = load_wine()
X = dataset["data"]
y = dataset["target"]


# In[5]:


# create k-fold cross validation dataset
kfold = KFold(n_splits=5,              # 5-fold
              random_state=101,       # fix random seed
              shuffle=True)           # get fold after shuffle


# In[6]:


acc = []
# iterate each fold by their index
for train_idx, test_idx in kfold.split(X):          # iterates index of each fold
    X_train, X_test = X[train_idx], X[test_idx]     # get input for current fold
    y_train, y_test = y[train_idx], y[test_idx]     # get target for current fold
    
    svm = SVC(C=1.0)                                # create linear svm model
    svm.fit(X_train, y_train)                       # train on current fold
    y_pred = svm.predict(X_test)                    # predict on current fold
    acc.append(accuracy_score(y_test, y_pred))      # accuracy score of current fold


# In[8]:


print(sum(acc) / len(acc))                          # average accuracy of the model


# In[10]:


acc = []
# iterate each fold by their index
for train_idx, test_idx in kfold.split(X):          # iterates index of each fold
    X_train, X_test = X[train_idx], X[test_idx]     # get input for current fold
    y_train, y_test = y[train_idx], y[test_idx]     # get target for current fold
    
    tree = DecisionTreeClassifier(ccp_alpha=0.2)    # create decision tree model
    tree.fit(X_train, y_train)                       # train on current fold
    y_pred = tree.predict(X_test)                    # predict on current fold
    acc.append(accuracy_score(y_test, y_pred))      # accuracy score of current fold


# In[11]:


print(sum(acc) / len(acc))                          # average accuracy of the model


# In[ ]:




