#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Zoomcamp
# 
# ## 1.8 Linear algebra refresher

# Plan:
# 
# * Vector operations
# * Multiplication
#     * Vector-vector multiplication
#     * Matrix-vector multiplication
#     * Matrix-matrix multiplication
# * Identity matrix
# * Inverse

# In[1]:


import numpy as np


# ## Vector operations

# In[2]:


u = np.array([2, 4, 5, 6])


# In[6]:


2 * u


# In[4]:


v = np.array([1, 0, 0, 2])


# In[5]:


u + v


# In[7]:


u * v


# ## Multiplication

# In[10]:


v.shape[0]


# In[11]:


def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
    
    n = u.shape[0]
    
    result = 0.0

    for i in range(n):
        result = result + u[i] * v[i]
    
    return result


# In[12]:


vector_vector_multiplication(u, v)


# In[13]:


u.dot(v)


# In[14]:


U = np.array([
    [2, 4, 5, 6],
    [1, 2, 1, 2],
    [3, 1, 2, 1],
])


# In[16]:


U.shape


# In[17]:


def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]
    
    num_rows = U.shape[0]
    
    result = np.zeros(num_rows)
    
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    
    return result


# In[18]:


matrix_vector_multiplication(U, v)


# In[19]:


U.dot(v)


# In[20]:


V = np.array([
    [1, 1, 2],
    [0, 0.5, 1], 
    [0, 2, 1],
    [2, 1, 0],
])


# In[21]:


def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]
    
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    
    result = np.zeros((num_rows, num_cols))
    
    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi
    
    return result


# In[22]:


matrix_matrix_multiplication(U, V)


# In[23]:


U.dot(V)


# ## Identity matrix

# In[25]:


I = np.eye(3)


# In[28]:


V


# In[27]:


V.dot(I)


# ## Inverse

# In[31]:


Vs = V[[0, 1, 2]]
Vs


# In[33]:


Vs_inv = np.linalg.inv(Vs)
Vs_inv


# In[34]:


Vs_inv.dot(Vs)


# ### Next 
# 
# Intro to Pandas

# In[ ]:




