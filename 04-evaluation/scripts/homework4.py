# %% Import libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

custom_seed = 1
df = pd.read_feather("../data/credit_cards.feather")
target = 'card'

# %% Binarizing the target variable

df.card = (df.card == 'yes').astype(int)


# %% Setting up the validation framework

def validation_framework(data=df, val=0.25, test=0.2, random_seed=custom_seed, target=target):
    """ Prepares the validation framework for a machine learning model
        :param str target: The target variable
        :param pd.DataFrame data: a dataframe
        :param float random_seed: a seed that controls reproducibility
        :return: the train, validation and test set, separated by features (X) and target (y)
        :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
    df_full_train, df_test = train_test_split(data, test_size=test, random_state=random_seed)
    df_train, df_val = train_test_split(df_full_train, test_size=val, random_state=random_seed)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    del df_train[target]
    del df_val[target]
    del df_test[target]

    return df_full_train, df_train, df_val, df_test, y_train, y_val, y_test


df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = validation_framework()

# %% Q1 - ROC AUC could also be used to evaluate feature importance of numerical variables.
# For each numerical variable, use it as score and compute AUC with the card variable.
# Use the training dataset for that.
# If your AUC is < 0.5, invert this variable by putting "-" in front
# Which numerical variable (among the following 4) has the highest AUC?
# reports, dependents, active, share
categorical = [x for x in df.select_dtypes(object).columns]
numerical = [df.select_dtypes(np.number).columns]

areas = dict()
for variable in ['reports', 'age', 'income', 'share', 'expenditure', 'dependents', 'months', 'majorcards', 'active']:
    model = LogisticRegression()
    model.fit(df_train[[variable]], y_train)
    y_pred = model.predict_proba(df_train[[variable]])[:, 1]
    areas[variable] = roc_auc_score(y_train, y_pred).round(4)

print(sorted(areas, key=areas.get, reverse=True)[:3])  # Expenditure has the largest AUC

# %% Training the model

# From now on, use these columns only:
features = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active", "owner",
            "selfemp"]


# Apply one-hot-encoding using `DictVectorizer` and train the logistic regression with these parameters:
# LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
def train_logistic_regression(df_train, y_train, C=1.0):
    dicts = df_train[features].to_dict(orient='records')

    dictionary_vectorizer = DictVectorizer(sparse=False)
    X_train = dictionary_vectorizer.fit_transform(dicts)

    logistic_model = LogisticRegression(solver='liblinear', C=C, max_iter=1_000)
    logistic_model.fit(X_train, y_train)

    return dictionary_vectorizer, logistic_model


def predict(df, dv, model):
    dicts = df[features].to_dict(orient='records')

    X = dv.fit_transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Training the Logistic Regression Model
dv, model = train_logistic_regression(df_train, y_train)
# Predicting the Validation Dataset
y_pred = predict(df_val, dv, model)
# Calculating the area under the ROC curve
roc_auc_score(y_val, y_pred)

# %% Q3 - Computing Precision and Recall for our Model.
# Evaluate the model on the validation dataset on all thresholds from 0.0 to 1.0 with step 0.01
# For each threshold, compute precision and recall
# Plot them
# At which threshold precision and recall curves intersect?
thresholds = np.arange(0.0, 1.0, 0.01)
scores = []
for threshold in thresholds:
    real_positive = (y_val == 1)
    real_negative = (y_val == 0)

    predict_positive = (y_pred >= threshold)
    predict_negative = (y_pred < threshold)

    tp = (predict_positive & real_positive).sum()
    tn = (predict_negative & real_negative).sum()

    fp = (predict_positive & real_negative).sum()
    fn = (predict_negative & real_positive).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    scores.append((threshold, precision, recall))

scores = pd.DataFrame(scores, columns={'threshold', 'precision', 'recall'})
# Plotting the data
plt.plot(scores.threshold, scores.precision, label='Precision')  # Thresholds vs Precision
plt.plot(scores.threshold, scores.recall, label='Recall')  # Thresholds vs Recall
plt.legend()
plt.show()


# Both curves intersect at about the 0.3 threshold

# %% Q4- Computing the F1 Score
# Precision and recall are conflicting - when one grows, the other goes down.
# That's why they are often combined into the F1 score - a metrics that takes into account both
def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


scores['f1_score'] = f1_score(scores.precision, scores.recall)

# At which threshold F1 is maximal?
maximal_f1_score = scores.nlargest(n=1, columns='f1_score')[
    ['threshold', 'f1_score']]  # F1 is maximal at the 0.35 threshold
print(maximal_f1_score)

# Plotting the f1_score across all the thresholds
plt.plot(scores.threshold, scores.f1_score, label='f1_score')
plt.legend()
plt.show()  # F1 is maximal at the 0.35 threshold

# %% Q5 - Use the `KFold` class from sklearn to evaluate our model on 5 different folds:
# Iterate over different folds of df_full_train
# Split the data into train and validation
# Train the model on train with these parameters: LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# Use AUC to evaluate the model on validation
# How large is standard devidation of the AUC scores across different folds?

kFold = KFold(n_splits=5, shuffle=True, random_state=custom_seed)
scores = []

for train_idx, val_idx in kFold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card.values
    y_val = df_val.card.values

    dv, model = train_logistic_regression(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

std_folds = np.std(scores)
print(std_folds)

# %% Q6 - Now let's use 5-Fold cross-validation to find the best parameter C
# Iterate over the following C values: [0.01, 0.1, 1, 10]
# Initialize KFold with the same parameters as previously
# Use these parametes for the model: LogisticRegression(solver='liblinear', C=C, max_iter=1000)
# Compute the mean score as well as the std (round the mean and std to 3 decimal digits)
# Which C leads to the best mean score?

C = [0.01, 0.1, 1, 10]
fold_stats = pd.DataFrame()

for c in [0.01, 0.1, 1, 10]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.card.values
        y_val = df_val.card.values

        dv, model = train_logistic_regression(df_train, y_train, C=c)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    data = pd.DataFrame([{'param': c, 'mean': np.mean(scores).round(3), 'sd': np.std(scores).round(3)}])
    fold_stats = pd.concat([fold_stats, data])

print(fold_stats)  # The best c is the default value 1
