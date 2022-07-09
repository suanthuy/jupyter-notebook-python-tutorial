#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint


# In[2]:


dataset = load_wine()
X = dataset["data"]
y = dataset["target"]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,                     # input features
                                                    y,                     # output target
                                                    test_size=0.3,         # train:test = 70:30
                                                    random_state=101,      # fixed random seed
                                                    stratify=y)            # keep target ratio in both splits


# In[4]:


# set PDF(probability density function) or PMF (probability mass function) for each continous parameter
dists = {"C": uniform(loc=0, scale=1.0),      # C parameter is drawn from uniform distribution (0.0 ~ 1.0)
         "gamma": uniform(loc=0, scale=1.0),  # gamma parameter is drawn from uniform distribution (0.0 ~ 1.0)
         "kernel": ["linear", "rbf"]}         # kernel parameter is drawn from {"linear", "rbf"} set

cv = RandomizedSearchCV(SVC(),                 # target model is set to linear svm
                        dists,                 # set PDFs of each parameter
                        scoring="accuracy",    # target measure is set to accuracy
                        n_iter=1000,           # number of experiments
                        cv=7,                  # 7-fold cross validation
                        n_jobs=-1)


# In[5]:


cv.fit(X_train, y_train)     # train n_iter times using randomly drawn parameters
print(cv.best_params_)       # best parameters


# In[7]:


print(classification_report(y_test, cv.predict(X_test)))


# In[8]:


# set PDF(probability density function) or PMF (probability mass function) for each continous parameter
dists = {"max_depth": randint(1, 101),               # max_depth parameter is drawn from uniform distribution (1 ~ 100)
         "ccp_alpha": uniform(loc=0.0, scale=10.0)}  # ccp_alpha parameter is drawn from uniform distribution (0.0 ~ 1.0)

cv = RandomizedSearchCV(DecisionTreeClassifier(),                 # target model is set to linear svm
                        dists,                 # set PDFs of each parameter
                        scoring="accuracy",    # target measure is set to accuracy
                        n_iter=1000,           # number of experiments
                        cv=7,                  # 7-fold cross validation
                        n_jobs=-1)


# In[9]:


cv.fit(X_train, y_train)     # train n_iter times using randomly drawn parameters
print(cv.best_params_)       # best parameters


# In[10]:


print(classification_report(y_test, cv.predict(X_test)))


# In[ ]:




