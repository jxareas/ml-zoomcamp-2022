#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Zoomcamp
# 
# ## 1.9 Introduction to Pandas
# 
# Plan:
# 
# * Data Frames
# * Series
# * Index
# * Accessing elements
# * Element-wise operations
# * Filtering
# * String operations
# * Summarizing operations
# * Missing values
# * Grouping
# * Getting the NumPy arrays

# In[4]:


import numpy as np
import pandas as pd


# ## DataFrames

# In[5]:


data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]


# In[8]:


df = pd.DataFrame(data, columns=columns)


# In[9]:


df


# In[10]:


data = [
    {
        "Make": "Nissan",
        "Model": "Stanza",
        "Year": 1991,
        "Engine HP": 138.0,
        "Engine Cylinders": 4,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "sedan",
        "MSRP": 2000
    },
    {
        "Make": "Hyundai",
        "Model": "Sonata",
        "Year": 2017,
        "Engine HP": None,
        "Engine Cylinders": 4,
        "Transmission Type": "AUTOMATIC",
        "Vehicle_Style": "Sedan",
        "MSRP": 27150
    },
    {
        "Make": "Lotus",
        "Model": "Elise",
        "Year": 2010,
        "Engine HP": 218.0,
        "Engine Cylinders": 4,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "convertible",
        "MSRP": 54990
    },
    {
        "Make": "GMC",
        "Model": "Acadia",
        "Year": 2017,
        "Engine HP": 194.0,
        "Engine Cylinders": 4,
        "Transmission Type": "AUTOMATIC",
        "Vehicle_Style": "4dr SUV",
        "MSRP": 34450
    },
    {
        "Make": "Nissan",
        "Model": "Frontier",
        "Year": 2017,
        "Engine HP": 261.0,
        "Engine Cylinders": 6,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "Pickup",
        "MSRP": 32340
    }
]


# In[12]:


df = pd.DataFrame(data)
df


# In[14]:


df.head(n=2)


# In[ ]:





# ## Series

# In[18]:


df.Engine HP


# In[19]:


df['Engine HP']


# In[20]:


df[['Make', 'Model', 'MSRP']]


# In[23]:


df['id'] = [1, 2, 3, 4, 5]


# In[26]:


df['id'] = [10, 20, 30, 40, 50]


# In[27]:


df


# In[28]:


del df['id']


# In[29]:


df


# ## Index
# 

# In[30]:


df.index


# In[32]:


df.Make.index


# In[37]:


df.index = ['a', 'b', 'c', 'd', 'e']


# In[38]:


df


# In[42]:


df.iloc[[1, 2, 4]]


# In[47]:


df = df.reset_index(drop=True)


# In[48]:


df


# ## Accessing elements

# In[ ]:





# ## Element-wise operations

# In[51]:


df['Engine HP'] * 2


# In[53]:


df['Year'] >= 2015


# ## Filtering

# In[55]:


df[
    df['Make'] == 'Nissan'
]


# In[56]:


df[
    (df['Make'] == 'Nissan') & (df['Year'] >= 2015)
]


# ## String operations

# In[68]:


'machine learning zoomcamp'.replace(' ', '_')


# In[67]:


df['Vehicle_Style'].str.lower()


# In[72]:


df['Vehicle_Style'] = df['Vehicle_Style'].str.replace(' ', '_').str.lower()


# In[74]:


df


# ## Summarizing operations

# In[81]:


df.describe().round(2)


# In[85]:


df.nunique()


# ## Missing values
# 

# In[87]:


df.isnull().sum()


# ## Grouping
# 

# ```
# SELECT 
#     transmission_type,
#     AVG(MSRP)
# FROM
#     cars
# GROUP BY
#     transmission_type
# ```

# In[90]:


df.groupby('Transmission Type').MSRP.max()


# ## Getting the NumPy arrays

# In[93]:


df.MSRP.values


# In[95]:


df.to_dict(orient='records')


# In[ ]:




