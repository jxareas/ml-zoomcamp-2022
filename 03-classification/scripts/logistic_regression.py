#!/usr/bin/env python
# coding: utf-8

# # 3. Machine Learning for Classification
# 
# We'll use logistic regression to predict churn
# 
# 
# ## 3.1 Churn prediction project
# 
# * Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
# * https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv
# 

# ## 3.2 Data preparation
# 
# * Download the data, read it with pandas
# * Look at the data
# * Make column names and values look uniform
# * Check if all the columns read correctly
# * Check if the churn variable needs any preparation

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[2]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[3]:


get_ipython().system('wget $data -O data-week-3.csv ')


# In[4]:


df = pd.read_csv('data-week-3.csv')
df.head()


# In[5]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[6]:


df.head().T


# In[7]:


tc = pd.to_numeric(df.totalcharges, errors='coerce')


# In[8]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')


# In[9]:


df.totalcharges = df.totalcharges.fillna(0)


# In[10]:


df.churn.head()


# In[11]:


df.churn = (df.churn == 'yes').astype(int)


# ## 3.3 Setting up the validation framework
# 
# * Perform the train/validation/test split with Scikit-Learn

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[14]:


len(df_train), len(df_val), len(df_test)


# In[15]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[16]:


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


# ## 3.4 EDA
# 
# * Check missing values
# * Look at the target variable (churn)
# * Look at numerical and categorical variables

# In[17]:


df_full_train = df_full_train.reset_index(drop=True)


# In[18]:


df_full_train.isnull().sum()


# In[19]:


df_full_train.churn.value_counts(normalize=True)


# In[20]:


df_full_train.churn.mean()


# In[21]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[22]:


categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[23]:


df_full_train[categorical].nunique()


# ## 3.5 Feature importance: Churn rate and risk ratio
# 
# Feature importance analysis (part of EDA) - identifying which features affect our target variable
# 
# * Churn rate
# * Risk ratio
# * Mutual information - later

# #### Churn rate

# In[24]:


df_full_train.head()


# In[25]:


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female


# In[26]:


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male


# In[27]:


global_churn = df_full_train.churn.mean()
global_churn


# In[28]:


global_churn - churn_female


# In[29]:


global_churn - churn_male


# In[30]:


df_full_train.partner.value_counts()


# In[31]:


churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner


# In[32]:


global_churn - churn_partner


# In[33]:


churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner


# In[34]:


global_churn - churn_no_partner


# #### Risk ratio

# In[35]:


churn_no_partner / global_churn


# In[36]:


churn_partner / global_churn


# ```
# SELECT
#     gender,
#     AVG(churn),
#     AVG(churn) - global_churn AS diff,
#     AVG(churn) / global_churn AS risk
# FROM
#     data
# GROUP BY
#     gender;
# ```

# In[37]:


from IPython.display import display


# In[38]:


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()
    print()


# ## 3.6 Feature importance: Mutual information
# 
# Mutual information - concept from information theory, it tells us how much 
# we can learn about one variable if we know the value of another
# 
# * https://en.wikipedia.org/wiki/Mutual_information

# In[39]:


from sklearn.metrics import mutual_info_score


# In[40]:


mutual_info_score(df_full_train.churn, df_full_train.contract)


# In[41]:


mutual_info_score(df_full_train.gender, df_full_train.churn)


# In[42]:


mutual_info_score(df_full_train.contract, df_full_train.churn)


# In[43]:


mutual_info_score(df_full_train.partner, df_full_train.churn)


# In[44]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[45]:


mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)


# ## 3.7 Feature importance: Correlation
# 
# How about numerical columns?
# 
# * Correlation coefficient - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

# In[46]:


df_full_train.tenure.max()


# In[47]:


df_full_train[numerical].corrwith(df_full_train.churn).abs()


# In[48]:


df_full_train[df_full_train.tenure <= 2].churn.mean()


# In[49]:


df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()


# In[50]:


df_full_train[df_full_train.tenure > 12].churn.mean()


# In[51]:


df_full_train[df_full_train.monthlycharges <= 20].churn.mean()


# In[52]:


df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean()


# In[53]:


df_full_train[df_full_train.monthlycharges > 50].churn.mean()


# ## 3.8 One-hot encoding
# 
# * Use Scikit-Learn to encode categorical features

# In[54]:


from sklearn.feature_extraction import DictVectorizer


# In[55]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# ## 3.9 Logistic regression
# 
# * Binary classification
# * Linear vs logistic regression

# In[56]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[57]:


z = np.linspace(-7, 7, 51)


# In[58]:


sigmoid(10000)


# In[59]:


plt.plot(z, sigmoid(z))


# In[60]:


def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result


# In[61]:


def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        score = score + xi[j] * w[j]
        
    result = sigmoid(score)
    return result


# ## 3.10 Training logistic regression with Scikit-Learn
# 
# * Train a model with Scikit-Learn
# * Apply it to the validation dataset
# * Calculate the accuracy

# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


model = LogisticRegression(solver='lbfgs')
# solver='lbfgs' is the default solver in newer version of sklearn
# for older versions, you need to specify it explicitly
model.fit(X_train, y_train)


# In[64]:


model.intercept_[0]


# In[65]:


model.coef_[0].round(3)


# In[66]:


y_pred = model.predict_proba(X_val)[:, 1]


# In[67]:


churn_decision = (y_pred >= 0.5)


# In[68]:


(y_val == churn_decision).mean()


# In[69]:


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val


# In[70]:


df_pred['correct'] = df_pred.prediction == df_pred.actual


# In[71]:


df_pred.correct.mean()


# In[72]:


churn_decision.astype(int)


# ## 3.11 Model interpretation
# 
# * Look at the coefficients
# * Train a smaller model with fewer features

# In[73]:


a = [1, 2, 3, 4]
b = 'abcd'


# In[74]:


dict(zip(a, b))


# In[75]:


dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))


# In[76]:


small = ['contract', 'tenure', 'monthlycharges']


# In[77]:


df_train[small].iloc[:10].to_dict(orient='records')


# In[78]:


dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')


# In[79]:


dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)


# In[80]:


dv_small.get_feature_names()


# In[81]:


X_train_small = dv_small.transform(dicts_train_small)


# In[82]:


model_small = LogisticRegression(solver='lbfgs')
model_small.fit(X_train_small, y_train)


# In[83]:


w0 = model_small.intercept_[0]
w0


# In[84]:


w = model_small.coef_[0]
w.round(3)


# In[85]:


dict(zip(dv_small.get_feature_names(), w.round(3)))


# In[86]:


-2.47 + (-0.949) + 30 * 0.027 + 24 * (-0.036)


# In[87]:


sigmoid(_)


# ## 3.12 Using the model

# In[88]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')


# In[89]:


dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)


# In[90]:


y_full_train = df_full_train.churn.values


# In[91]:


model = LogisticRegression(solver='lbfgs')
model.fit(X_full_train, y_full_train)


# In[92]:


dicts_test = df_test[categorical + numerical].to_dict(orient='records')


# In[93]:


X_test = dv.transform(dicts_test)


# In[94]:


y_pred = model.predict_proba(X_test)[:, 1]


# In[95]:


churn_decision = (y_pred >= 0.5)


# In[96]:


(churn_decision == y_test).mean()


# In[97]:


y_test


# In[98]:


customer = dicts_test[-1]
customer


# In[99]:


X_small = dv.transform([customer])


# In[100]:


model.predict_proba(X_small)[0, 1]


# In[101]:


y_test[-1]


# ## 3.13 Summary
# 
# * Feature importance - risk, mutual information, correlation
# * One-hot encoding can be implemented with `DictVectorizer`
# * Logistic regression - linear model like linear regression
# * Output of log reg - probability
# * Interpretation of weights is similar to linear regression

# ## 3.14 Explore more
# 
# More things
# 
# * Try to exclude least useful features
# 
# 
# Use scikit-learn in project of last week
# 
# * Re-implement train/val/test split using scikit-learn in the project from the last week
# * Also, instead of our own linear regression, use `LinearRegression` (not regularized) and `RidgeRegression` (regularized). Find the best regularization parameter for Ridge
# 
# Other projects
# 
# * Lead scoring - https://www.kaggle.com/ashydv/leads-dataset
# * Default prediction - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# 
# 
