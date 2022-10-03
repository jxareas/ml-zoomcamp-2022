#!/usr/bin/env python
# coding: utf-8

# ## 2. Machine Learning for Regression
# 

# In[1]:


import pandas as pd
import numpy as np


# ## 2.2 Data preparation

# In[2]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'


# In[5]:


get_ipython().system('wget $data ')


# In[5]:


df = pd.read_csv('data.csv')


# In[6]:


df.columns = df.columns.str.lower().str.replace(' ', '_')


# In[7]:


df['make'].str.lower().str.replace(' ', '_')


# In[8]:


strings = list(df.dtypes[df.dtypes == 'object'].index)
strings


# In[9]:


for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[10]:


df.dtypes


# ## 2.3 Exploratory data analysis

# In[11]:


for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()


# In[12]:


df


# Distribution of price

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


sns.histplot(df.msrp, bins=50)


# In[15]:


sns.histplot(df.msrp[df.msrp < 100000], bins=50)


# In[16]:


np.log1p([0, 1, 10, 1000, 100000])


# In[17]:


np.log([0 + 1, 1+ 1, 10 + 1, 1000 + 1, 100000])


# In[18]:


price_logs = np.log1p(df.msrp)


# In[19]:



sns.histplot(price_logs, bins=50)


# Missing values

# In[20]:


df.isnull().sum()


# ## 2.4 Setting up the validation framework

# Let's draw it

# In[21]:


n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test


# In[22]:


n


# In[23]:


n_val, n_test, n_train


# In[24]:


df.iloc[[10, 0, 3, 5]]


# In[25]:


df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]


# In[26]:


idx = np.arange(n)


# In[27]:


np.random.seed(2)
np.random.shuffle(idx)


# In[28]:


df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]


# In[29]:


df_train.head()


# In[30]:


len(df_train), len(df_val), len(df_test)


# In[31]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[32]:


y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)


# In[33]:


del df_train['msrp']
del df_val['msrp']
del df_test['msrp']


# In[34]:


len(y_train)


# ## 2.5 Linear regression

# draw

# In[35]:


df_train.iloc[10]


# In[ ]:





# In[36]:


xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]


# In[37]:


def linear_regression(xi):
    n = len(xi)

    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred


# In[38]:


xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]


# In[39]:


linear_regression(xi)


# In[40]:


np.expm1(12.312)


# In[41]:


np.log1p(222347.2221101062)


# ## 2.6 Linear regression vector form

# In[42]:


def dot(xi, w):
    n = len(xi)
    
    res = 0.0
    
    for j in range(n):
        res = res + xi[j] * w[j]
    
    return res


# In[43]:


def linear_regression(xi):
    return w0 + dot(xi, w)


# In[44]:


w_new = [w0] + w


# In[45]:


w_new


# In[46]:


def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)


# In[47]:


linear_regression(xi)


# In[48]:


w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w


# In[49]:


x1  = [1, 148, 24, 1385]
x2  = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
X = np.array(X)
X


# In[50]:


def linear_regression(X):
    return X.dot(w_new)


# In[51]:


linear_regression(X)


# ## 2.7 Training a linear regression model

# In[52]:


def train_linear_regression(X, y):
    pass


# In[53]:


X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]

X = np.array(X)
X


# In[54]:


ones = np.ones(X.shape[0])
ones


# In[55]:


X = np.column_stack([ones, X])


# In[56]:


y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]


# In[57]:


XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
w_full = XTX_inv.dot(X.T).dot(y)


# In[ ]:





# In[ ]:





# In[58]:


w0 = w_full[0]
w = w_full[1:]


# In[59]:


w0, w


# In[60]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


# In[61]:


train_linear_regression(X, y)


# ## 2.8 Car price baseline model

# In[62]:


df_train.columns


# In[63]:


base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)


# In[64]:


w0


# In[65]:


w


# In[ ]:





# In[66]:


sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_train, color='blue', alpha=0.5, bins=50)


# ## 2.9 RMSE

# In[67]:


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


# In[68]:


rmse(y_train, y_pred)


# ## 2.10 Validating the model

# In[69]:


def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# In[70]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# ## 2.11 Simple feature engineering

# In[71]:


def prepare_X(df):
    df = df.copy()
    
    df['age'] = 2017 - df['year']
    features = base + ['age']
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


# In[72]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# In[73]:


sns.histplot(y_pred, label='prediction', color='red', alpha=0.5, bins=50)
sns.histplot(y_val, label='target', color='blue',  alpha=0.5, bins=50)
plt.legend()


# ## 2.12 Categorical variables

# In[74]:


categorical_columns = [
    'make', 'model', 'engine_fuel_type', 'driven_wheels', 'market_category',
    'vehicle_size', 'vehicle_style']

categorical = {}

for c in categorical_columns:
    categorical[c] = list(df_train[c].value_counts().head().index)


# In[75]:


def prepare_X(df):
    df = df.copy()
    
    df['age'] = 2017 - df['year']
    features = base + ['age']

    for v in [2, 3, 4]:
        df['num_doors_%d' % v] = (df.number_of_doors == v).astype(int)
        features.append('num_doors_%d' % v)

    for name, values in categorical.items():
        for value in values:
            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


# In[76]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# In[77]:


w0, w


# ## 2.13 Regularization

# In[78]:


X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.00000001],
]

X = np.array(X)
X


# In[79]:


y= [1, 2, 3, 1, 2, 3]


# In[80]:


XTX = X.T.dot(X)
XTX


# In[81]:


XTX_inv = np.linalg.inv(XTX)


# In[82]:


XTX_inv


# In[83]:


XTX_inv.dot(X.T).dot(y)


# In[84]:


XTX = [
    [1, 2, 2],
    [2, 1, 1.0000001],
    [2, 1.0000001, 1]
]

XTX = np.array(XTX)


# In[85]:


np.linalg.inv(XTX)


# In[86]:


XTX = XTX + 0.01 * np.eye(3)


# In[87]:


np.linalg.inv(XTX)


# In[88]:


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


# In[89]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# ## 2.14 Tuning the model

# In[90]:


for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    
    print(r, w0, score)


# In[91]:


r = 0.001
X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=r)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
score = rmse(y_val, y_pred)
score


# ## 2.15 Using the model

# In[92]:


df_full_train = pd.concat([df_train, df_val])


# In[93]:


df_full_train = df_full_train.reset_index(drop=True)


# In[94]:


X_full_train = prepare_X(df_full_train)


# In[95]:


X_full_train


# In[96]:


y_full_train = np.concatenate([y_train, y_val])


# In[97]:


w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)


# In[98]:


X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score


# In[99]:


car = df_test.iloc[20].to_dict()
car


# In[100]:


df_small = pd.DataFrame([car])
df_small


# In[101]:


X_small = prepare_X(df_small)


# In[102]:


y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0]
y_pred


# In[103]:


np.expm1(y_pred)


# In[104]:


np.expm1(y_test[20])


# ## 2.16 Next steps

# * We included only 5 top features. What happens if we include 10?
# 
# Other projects
# 
# * Predict the price of a house - e.g. boston dataset
# * https://archive.ics.uci.edu/ml/datasets.php?task=reg
# * https://archive.ics.uci.edu/ml/datasets/Student+Performance

# ## 2.17 Summary
# 
# * EDA - looking at data, finding missing values
# * Target variable distribution - long tail => bell shaped curve
# * Validation framework: train/val/test split (helped us detect problems)
# * Normal equation - not magic, but math
# * Implemented it with numpy
# * RMSE to validate our model
# * Feature engineering: age, categorical features
# * Regularization to fight numerical instability

# In[ ]:




