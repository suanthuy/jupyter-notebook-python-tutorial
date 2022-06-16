#!/usr/bin/env python
# coding: utf-8

# # 1. Seaborn Package

# ## 1.1 Histplot

# In[1]:


get_ipython().system('pip install seaborn')


# In[2]:


import seaborn as sns


# In[3]:


df = sns.load_dataset("penguins")
df.head()


# In[5]:


df


# In[4]:


sns.histplot(df["bill_length_mm"])


# In[6]:


sns.histplot(df["bill_length_mm"], bins=25) # control the number of bins (number of bars)


# In[7]:


sns.histplot(df, x="bill_length_mm", bins=25, kde=True) # kde (kernel density estimate) plot


# ## 1.2 Jointplot

# In[8]:


import seaborn as sns


# In[9]:


df = sns.load_dataset("penguins")
df.head()


# In[10]:


sns.jointplot(data=df, x="bill_length_mm", y="bill_depth_mm") # scatter plot


# In[11]:


sns.jointplot(data=df, x="bill_length_mm", y="bill_depth_mm", kind="hex") # hexagonal histogram


# In[12]:


sns.jointplot(data=df, x="bill_length_mm", y="bill_depth_mm", kind="reg") # regression plot


# In[14]:


sns.jointplot(data=df, x="bill_length_mm", y="bill_depth_mm", kind="kde")


# ## 1.3 Pairplot

# In[15]:


import seaborn as sns


# In[17]:


df = sns.load_dataset("penguins")
df.head()


# In[18]:


sns.pairplot(data=df) # default pairplot (diagonal, histplot, others: scatterplot)


# In[19]:


sns.pairplot(data=df, hue="species") # hue to split categories


# ## 1.4 Barplot

# In[21]:


import numpy as np
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[22]:


sns.barplot(data=df, x="sex", y="bill_length_mm") # default estimator = average


# In[23]:


sns.barplot(data=df, x="sex", y="bill_length_mm", estimator=np.std) # estimator is set to std (standard deviation)


# In[24]:


sns.barplot(data=df, x="sex", y="bill_length_mm", hue="species")


# ## 1.5 Countplot

# In[27]:


import numpy as np
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[28]:


sns.countplot(data=df, x="species")


# In[29]:


sns.countplot(data=df, x="species", hue="sex")


# ## 1.6 Boxplot

# In[31]:


import numpy as np
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[33]:


sns.boxplot(data=df, x="species", y="body_mass_g")


# In[34]:


sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# ## 1.7 Violinplot -> use to understand for you, not for the other

# In[35]:


import numpy as np
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[36]:


sns.violinplot(data=df, x="species", y="body_mass_g")


# In[37]:


sns.violinplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[38]:


sns.violinplot(data=df, x="species", y="body_mass_g", hue="sex", split=True)


# ## 1.8 Stripplot

# In[39]:


import numpy as np
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[40]:


sns.stripplot(data=df, x="species", y="body_mass_g")


# In[41]:


sns.stripplot(data=df, x="species", y="body_mass_g", hue="sex", alpha=0.5)


# In[42]:


sns.stripplot(data=df, x="species", y="body_mass_g", jitter=False, alpha=0.2) # jitter: False (no horizontal)


# ## 1.9 Swarmplot

# In[43]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[44]:


sns.swarmplot(data=df, x="species", y="body_mass_g")


# In[45]:


sns.swarmplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[46]:


sns.swarmplot(data=df, x="species", y="body_mass_g", hue="sex", dodge=True)


# ## 1.10 Heatmap

# In[47]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[48]:


corr = df.corr()
sns.heatmap(corr) # basic heatmap


# In[53]:


sns.heatmap(corr, annot=True, cmap="RdYlBu_r")


# # 1.11 Pairgrid

# In[54]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[59]:


g = sns.PairGrid(df)
g.map_diag(sns.histplot) # set diagonal plots to histplot
g.map_upper(sns.scatterplot) # set upper plots to scatterplot
g.map_lower(sns.kdeplot) # set lower plots to kdeplot


# ## 1.12 FacetGrid

# In[60]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[61]:


g = sns.FacetGrid(data=df, col="species", row="sex")
g.map(sns.histplot, "body_mass_g", kde=True)


# ## 1.13 Seaborn Styles

# In[62]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")
df


# In[63]:


# white, dark, whitegrid, darkgrid, ticks
sns.set_style("dark")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[64]:


sns.set_style("darkgrid")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[65]:


sns.set_style("white")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[66]:


sns.set_style("whitegrid")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[67]:


sns.set_style("ticks")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# ## 1.14 Seaborn Styles - Despine

# In[68]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")


# In[69]:


sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")
sns.despine() # default: right=True, top=True


# In[70]:


sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")
sns.despine(left=True, bottom=True) # default: right=True, top=True


# ## 1.15 Seaborn Styles - Contexts

# In[71]:


import numpy as np 
import seaborn as sns

df = sns.load_dataset("penguins")


# In[72]:


sns.set_context("paper")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[73]:


sns.set_context("notebook")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[74]:


sns.set_context("talk")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[75]:


sns.set_context("poster")
sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex")


# In[78]:


sns.dogplot()


# In[ ]:




