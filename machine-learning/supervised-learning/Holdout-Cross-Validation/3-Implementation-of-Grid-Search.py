#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report


# In[3]:


dataset = load_wine()
X = dataset["data"]
y = dataset["target"]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X,                     # input features
                                                    y,                     # output target
                                                    test_size=0.3,         # train:test = 70:30
                                                    random_state=101,      # fixed random seed
                                                    stratify=y)            # keep target ratio in both splits


# In[6]:


# search grid for each parameter
params = {"C": [0.01, 0.05, 0.2, 0.5, 0.8, 1.0],     # set search space for C parameter
          "gamma": ["scale", 0.2, 0.5, 1.0],         # set search space for gamma parameter
          "kernel": ["linear", "rbf"]                # set search space for kernel parameter
         }

cv = GridSearchCV(SVC(),                             # target model is set to linear svm
                  param_grid=params,                 # set parameter grid
                  scoring="accuracy",                # target measure is set to accuracy
                  cv=7,                              # 7-fold cross validation
                  n_jobs=-1)


# In[7]:


cv.fit(X_train, y_train)  # train all parameters and choose best model
print(cv.best_params_)    # best parameters


# In[8]:


print(classification_report(y_test, cv.predict(X_test)))


# In[9]:


# search grid for each parameter
params = {"max_depth": [5,10,20,50,100],    # set search space for max_depth parameter
          "ccp_alpha": [0.0, 0.2, 0.5, 0.8, 1.0, 2.0]}  # set search space for ccp_alpha parameter

cv = GridSearchCV(DecisionTreeClassifier(),      # target model is set to decision tree
                  param_grid=params,             # set parameter grid
                  scoring="accuracy",            # target measure is set to accuracy
                  cv=7,                          # 7-fold cross validation
                  n_jobs=-1)


# In[10]:


cv.fit(X_train, y_train)    # train all parameters and choose best model
print(cv.best_params_)      # best parameter


# In[11]:


print(classification_report(y_test, cv.predict(X_test)))


# In[ ]:




