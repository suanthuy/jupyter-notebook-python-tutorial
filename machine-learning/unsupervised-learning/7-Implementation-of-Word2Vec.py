#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install gensim')


# In[4]:


from gensim.models import Word2Vec
import gensim.downloader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[5]:


google_news = gensim.downloader.load("word2vec-google-news-300")   # download google news word2vec model


# In[ ]:




