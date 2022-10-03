#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Zoomcamp
# 
# 
# ## 1.7 Introduction to NumPy
# 
# 
# Plan:
# 
# * Creating arrays
# * Multi-dementional arrays
# * Randomly generated arrays
# * Element-wise operations
#     * Comparison operations
#     * Logical operations
# * Summarizing operations

# In[7]:


import numpy as np


# In[8]:


np


# ## Creating arrays
# 

# In[10]:


np.zeros(10)


# In[11]:


np.ones(10)


# In[12]:


np.full(10, 2.5)


# In[15]:


a = np.array([1, 2, 3, 5, 7, 12])
a


# In[17]:


a[2] = 10


# In[18]:


a


# In[20]:


np.arange(3, 10)


# In[24]:


np.linspace(0, 100, 11)


# ## Multi-dementional arrays
# 

# In[25]:


np.zeros((5, 2))


# In[27]:


n = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])


# In[29]:


n[0, 1] = 20


# In[30]:


n


# In[34]:


n[2] = [1, 1, 1]


# In[35]:


n


# In[39]:


n[:, 2] = [0, 1, 2]


# In[40]:


n


# ## Randomly generated arrays
# 

# In[51]:


np.random.seed(2)
100 * np.random.rand(5, 2)


# In[50]:


np.random.seed(2)
np.random.randn(5, 2)


# In[52]:


np.random.seed(2)
np.random.randint(low=0, high=100, size=(5, 2))


# ## Element-wise operations
# 

# In[53]:


a = np.arange(5)
a


# In[62]:


b = (10 + (a * 2)) ** 2 / 100


# In[65]:


b


# In[68]:


a / b + 10


# ## Comparison operations

# In[70]:


a


# In[69]:


a >= 2


# In[71]:


b


# In[72]:


a > b


# In[73]:


a[a > b]


# ## Summarizing operations

# In[75]:


a


# In[79]:


a.std()


# In[82]:


n.min()


# ### Next
# 
# Linear algebra refresher
