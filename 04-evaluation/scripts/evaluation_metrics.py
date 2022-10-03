#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '0')


# # 4. Evaluation Metrics for Classification
# 
# In the previous session we trained a model for predicting churn. How do we know if it's good?
# 
# 
# ## 4.1 Evaluation metrics: session overview 
# 
# * Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
# * https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv
# 
# 
# *Metric* - function that compares the predictions with the actual values and outputs a single number that tells how good the predictions are

# In[3]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[5]:


df = pd.read_csv('data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[6]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[7]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

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


# In[8]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)


# In[9]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()


# ## 4.2 Accuracy and dummy model
# 
# * Evaluate the model on different thresholds
# * Check the accuracy of dummy baselines

# In[16]:


len(y_val)


# In[19]:


(y_val == churn_decision).mean()


# In[18]:


1132/ 1409


# In[24]:


from sklearn.metrics import accuracy_score


# In[29]:


accuracy_score(y_val, y_pred >= 0.5)


# In[30]:


thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)


# In[24]:


plt.plot(thresholds, scores)


# In[34]:


from collections import Counter


# In[35]:


Counter(y_pred >= 1.0)


# In[39]:


1 - y_val.mean()


# ## 4.3 Confusion table
# 
# * Different types of errors and correct decisions
# * Arranging them in a table

# In[25]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[26]:


t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[27]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[28]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


# In[29]:


(confusion_matrix / confusion_matrix.sum()).round(2)


# ## 4.4 Precision and Recall

# In[30]:


p = tp / (tp + fp)
p


# In[31]:


r = tp / (tp + fn)
r


# ## 4.5 ROC Curves
# 
# ### TPR and FRP

# In[32]:


tpr = tp / (tp + fn)
tpr


# In[33]:


fpr = fp / (fp + tn)
fpr


# In[34]:


scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))


# In[35]:


columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)


# In[36]:


plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()


# ### Random model

# In[37]:


np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))


# In[38]:


((y_rand >= 0.5) == y_val).mean()


# In[47]:


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores


# In[40]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[41]:


plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.legend()


# ### Ideal model

# In[42]:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos


# In[43]:



y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal

y_ideal_pred = np.linspace(0, 1, len(y_val))


# In[44]:


1 - y_val.mean()


# In[45]:


accuracy_score(y_ideal, y_ideal_pred >= 0.726)


# In[51]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]


# In[52]:


plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
plt.legend()


# ### Putting everything together

# In[53]:


plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')

# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR random', color='grey')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR random', color='grey')

plt.legend()


# In[54]:


plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[55]:


from sklearn.metrics import roc_curve


# In[57]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[58]:


plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# ## 4.6 ROC AUC
# 
# * Area under the ROC curve - useful metric
# * Interpretation of AUC

# In[60]:


from sklearn.metrics import auc


# In[61]:


auc(fpr, tpr)


# In[62]:


auc(df_scores.fpr, df_scores.tpr)


# In[63]:


auc(df_ideal.fpr, df_ideal.tpr)


# In[68]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)


# In[65]:


from sklearn.metrics import roc_auc_score


# In[66]:


roc_auc_score(y_val, y_pred)


# In[70]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[73]:


import random


# In[82]:


n = 100000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n


# In[90]:


n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()


# ## 4.7 Cross-Validation
# 
# * Evaluating the same model on different subsets of data
# * Getting the average prediction and the spread within predictions

# In[121]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[123]:


dv, model = train(df_train, y_train, C=0.001)


# In[110]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[98]:


y_pred = predict(df_val, dv, model)


# In[99]:


from sklearn.model_selection import KFold


# In[100]:





# In[112]:


get_ipython().system('pip install tqdm')


# In[113]:


from tqdm.auto import tqdm


# In[129]:


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[133]:


scores


# In[131]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# ## 4.8 Summary
# 
# * Metric - a single number that describes the performance of a model
# * Accuracy - fraction of correct answers; sometimes misleading 
# * Precision and recall are less misleading when we have class inbalance
# * ROC Curve - a way to evaluate the performance at all thresholds; okay to use with imbalance
# * K-Fold CV - more reliable estimate for performance (mean + std)

# ## 4.9 Explore more
# 
# * Check the precision and recall of the dummy classifier that always predict "FALSE"
# * F1 score = 2 * P * R / (P + R)
# * Evaluate precision and recall at different thresholds, plot P vs R - this way you'll get the precision/recall curve (similar to ROC curve)
# * Area under the PR curve is also a useful metric
# 
# Other projects:
# 
# * Calculate the metrics for datasets from the previous week

# In[ ]:




