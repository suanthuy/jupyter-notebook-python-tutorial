#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score, plot_roc_curve, roc_auc_score, classification_report


# In[2]:


dataset = load_wine()
X = dataset["data"]
y = dataset["target"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,                  # input features
                                                    y,                  # output target
                                                    test_size=0.3,      # train: test = 70:30
                                                    random_state=101,   # fixed random seed
                                                    stratify=y)         # keep target ratio in both splits


# In[4]:


svm = SVC(C=1.0)                             # create linear svm model
svm.fit(X_train, y_train)                    # train model on train set
y_pred = svm.predict(X_test)                 # predict output on test set
print(classification_report(y_test, y_pred)) # print classification report
print("accuracy:", accuracy_score(y_test, y_pred))    # print accuracy score
print("f1_scores:", f1_score(y_test, y_pred, average="macro")) # print f1 score


# In[5]:


tree = DecisionTreeClassifier(ccp_alpha=0.2)            # create decision tree model w/ pruning
tree.fit(X_train, y_train)                              # train model on train set
y_pred = tree.predict(X_test)                           # predict output on test set
print(classification_report(y_test, y_pred)) # print classification report
print("accuracy:", accuracy_score(y_test, y_pred))    # print accuracy score
print("f1_scores:", f1_score(y_test, y_pred, average="macro")) # print f1 score


# In[ ]:




