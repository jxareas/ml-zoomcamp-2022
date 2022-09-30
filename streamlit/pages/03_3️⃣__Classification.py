from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
import pandas as pd
import numpy as np
import streamlit as st
from utils.streamlit_utils import top_question, br

custom_seed = 42
st.set_page_config(layout="wide")

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

st.title("Homework 3: Machine Learning for Classification")
st.markdown("""
This is a chapter that explores the theme of Classification in Machine Learning.
Algorithms in this chapter are implemented using the Scikit-Learn library.

Here are some of the topics covered by this chapter:

* Doing **Classification** via Logistic Regression
* Understanding and calculating the **Mutual Information** Score.
* Feature Elimination
""")
br()
st.subheader("Data Preview: ")

df = pd.read_csv(url)
st.dataframe(df)
br()

# %% Question 1

st.subheader("Data Preparation")
st.write("Filling the missing values and creating new columns via simple feature engineering.")
st.code("""
>>> # Select only the features from above and fill in the missing values with 0
>>> df[features] = df[features].fillna(0)
>>> # Create a new column rooms_per_household by dividing the column total_rooms by the column households from
>>> # the dataframe.
>>> df['rooms_per_household'] = df.total_rooms / df.households
>>> # Create a new column bedrooms_per_room by dividing the column total_bedrooms by the column total_rooms from
>>> # the dataframe.
>>> df['bedrooms_per_room'] = df.total_bedrooms / df.total_rooms
>>> # Create a new column population_per_household by dividing the column population by the column households from
>>> # the dataframe.
>>> df['population_per_household'] = df.population / df.households
""")
br()

# %% Question 1

top_question(1, "What is the most frequent observation (mode) for the column `ocean_proximity`?")
st.code("""
>>> df.ocean_proximity.mode()
""")
st.code(df.ocean_proximity.mode())
br()

# %% Setting up the validation framework
st.subheader("Setting up the Validation Framework")


def validation_framework(data=df, val=0.25, test=0.2, random_seed=custom_seed):
    """ Prepares the validation framework for a machine learning model
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

    y_train = df_train.median_house_value.values
    y_val = df_val.median_house_value.values
    y_test = df_test.median_house_value.values

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    return df_train, df_val, df_test, y_train, y_val, y_test


df_train, df_val, df_test, y_train, y_val, y_test = validation_framework()

st.code("""
def validation_framework(data=df, val=0.25, test=0.2, random_seed=custom_seed):
    df_full_train, df_test = train_test_split(data, test_size=test, random_state=random_seed)
    df_train, df_val = train_test_split(df_full_train, test_size=val, random_state=random_seed)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.median_house_value.values
    y_val = df_val.median_house_value.values
    y_test = df_test.median_house_value.values

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    return df_train, df_val, df_test, y_train, y_val, y_test


df_train, df_val, df_test, y_train, y_val, y_test = validation_framework()
""")
br()

# %% Question 2
top_question(2, "What are the two features that have the biggest correlation in this dataset?")
st.write("""
Create the correlation matrix for the numerical features of your train dataset.

In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.

What are the two features that have the biggest correlation in this dataset?
""")
st.code("""
>>> cor_matrix = df_train.corr(method='pearson')
>>> cor_matrix
>>> 
>>> def coefs(cor_matrix):
>>>     coeff_column = "coefficient"
>>> 
>>>     return cor_matrix \
>>>         .where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)) \
>>>         .stack() \
>>>         .reset_index() \
>>>         .rename(columns={0: coeff_column}) \
>>>         .sort_values(ascending=False, by=coeff_column, key=abs)

>>> # What are the two features that have the biggest correlation in this dataset?
>>> coefs(cor_matrix).head()  # Total Bedrooms & Households, with a coefficient of 0.97
""")


def coefs(cor_matrix):
    """ Gets the correlation coefficient entries from a correlation matrix
           :param pd.DataFrame data: the correlation matrix
           :return: correlation coefficients
           :rtype: pd.DataFrame
           """
    coeff_column = "coefficient"

    return cor_matrix \
        .where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)) \
        .stack() \
        .reset_index() \
        .rename(columns={0: coeff_column}) \
        .sort_values(ascending=False, by=coeff_column, key=abs)


coefficients = coefs(df_train.corr(method='pearson')).head()
st.code(coefficients)
br()

# %% Binarizing the `median_house_value` variable

st.subheader("Binarizing the `median_house_value` variable")
st.write("""
Defining a new target variable, which is 1 if `median_house_value` is greater than the median house value mean and 0 otherwise. 
This prepares the targets for a possible Logistic Regression model.
""")
st.code("""
# %% Make median_house_value binary
# We need to turn the median_house_value variable from numeric into binary.
# Let's create a variable above_average which is 1 if the median_house_value is above its mean value and 0 otherwise.

median_house_value_mean = df.median_house_value.mean()

# Binarizing the target
y_train = (y_train > median_house_value_mean).astype(int)
y_val = (y_val > median_house_value_mean).astype(int)
y_test = (y_test > median_house_value_mean).astype(int)
""")
median_house_value_mean = df.median_house_value.mean()

# Binarizing the target
y_train = (y_train > median_house_value_mean).astype(int)
y_val = (y_val > median_house_value_mean).astype(int)
y_test = (y_test > median_house_value_mean).astype(int)
br()

# %% Question 3

top_question(3, "Calculate the mutual informations score between `binarized price` and `ocean_proximity`.")
st.write("""
Calculate the mutual information score with the (binarized) price for the categorical variable that we have. Use the training set only.

What is the value of mutual information?

Round it to 2 decimal digits using round(score, 2)
""")
st.code("""
>>> mutual_information = mutual_info_score(df_train.ocean_proximity, y_train)
>>> print(round(mutual_information, 2))  # 0.1
""")
mutual_information = mutual_info_score(df_train.ocean_proximity, y_train)
st.code(round(mutual_information, 2))
br()

# %% One-Hot Encoding
st.subheader("One-hot Encoding")
st.write("Defining utilities to one-hot encode a dataframe using `DictVectorizer`.")
st.code("""
# %% Defining a function to one hot encode training data

def one_hot_encode(training_data):
    train_dicts = training_data.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    one_hot_encoded_data = dv.fit_transform(train_dicts)
    return one_hot_encoded_data
""")
st.code("""
# One hot encoding the features
X_train = one_hot_encode(df_train)
X_test = one_hot_encode(df_test)
X_val = one_hot_encode(df_val)
""")


def one_hot_encode(training_data):
    """ Prepares the validation framework for a machine learning model
        :param pd.DataFrame training_data: a dataframe
        :return: the dataframe with one-hot encoded data
        :rtype: np.ndarray
        """
    train_dicts = training_data.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    one_hot_encoded_data = dv.fit_transform(train_dicts)
    return one_hot_encoded_data


# One hot encoding the features
X_train = one_hot_encode(df_train)
X_test = one_hot_encode(df_test)
X_val = one_hot_encode(df_val)
br()

# %% Question 4
top_question(4, "Training a Logistic Regression")
st.write("""
Now let's train a logistic regression

Remember that we have one categorical variable ocean_proximity in the data. Include it using one-hot encoding. Fit the model on the training dataset.

To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:

`model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)`

Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
""")
st.code("""
>>> model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
>>> model.fit(X_train, y_train)
>>> 
>>> 
>>> def accuracy(model, X_validation, y_validation):
>>>     y_pred = model.predict(X_validation)
>>>     return (y_validation == y_pred).mean()
>>> 
>>> 
>>> global_accuracy = accuracy(model, X_val, y_val)
>>> print(round(global_accuracy, 2))  # 0.84
""")
model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)


def accuracy(model, X_validation, y_validation):
    y_pred = model.predict(X_validation)
    return (y_validation == y_pred).mean()


global_accuracy = accuracy(model, X_val, y_val)
st.code(round(global_accuracy, 2))

# %% Question 5

top_question(5, "Feature Elimination")
st.write("""
Let's find the least useful feature using the feature elimination technique.
Train a model with all these features (using the same parameters as in Q4). Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
For each feature, calculate the difference between the original accuracy and the accuracy without the feature. Which of following feature has the smallest difference?
""")
st.code("""
>>> diff_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
>>> diffs = dict()
>>> 
>>> for feature in diff_features:
>>>     features = df_train.columns.tolist()
>>>     features.remove(feature)
>>>     # Data Preparation
>>>     x_train = one_hot_encode(df_train[features])
>>>     # Model Training
>>>     model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
>>>     model.fit(x_train, y_train)
>>>     # Data Validation
>>>     x_val = one_hot_encode(df_val[features])
>>> 
>>>     # Computing the accuracy
>>>     acc = accuracy(model, x_val, y_val)
>>>     diffs[feature] = abs(global_accuracy - acc)
>>> 
>>> min_diff = {x: y for x, y in diffs.items() if y == min(diffs.values())}
>>> print(min_diff)  # total rooms
""")
diff_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
diffs = dict()

for feature in diff_features:
    features = df_train.columns.tolist()
    features.remove(feature)
    # Data Preparation
    x_train = one_hot_encode(df_train[features])
    # Model Training
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    # Data Validation
    x_val = one_hot_encode(df_val[features])

    # Computing the accuracy
    acc = accuracy(model, x_val, y_val)
    diffs[feature] = abs(global_accuracy - acc)

min_diff = {x: y for x, y in diffs.items() if y == min(diffs.values())}
st.code(min_diff)  # total rooms

# %% Question 6

top_question(6, "Select the multiplier with the smallest *RMSE*.")
st.write("""
For this question, we'll see how to use a linear regression model from Scikit-Learn
We'll need to use the original column 'median_house_value'. Apply the logarithmic transformation to this column.
Fit the Ridge regression model model = Ridge(alpha=a, solver="sag", random_state=42) on the training data.
This model has a parameter alpha.

Let's try the following values: [0, 0.01, 0.1, 1, 10].

Which of these alphas leads to the best RMSE on the validation set?

Round your RMSE scores to 3 decimal digits.

If there are multiple options, select the smallest alpha.
""")
st.code("""
>>> # %% Q6 - For this question, we'll see how to use a linear regression model from Scikit-Learn
>>> # We'll need to use the original column 'median_house_value'. Apply the logarithmic transformation to this column.
>>> 
>>> # Preparing the original validation framework with non-binarized target variable
>>> df_train, df_val, df_test, y_train, y_val, y_test = validation_framework()
>>> 
>>> # Applying logarithm transformation to the target
>>> y_train = np.log(y_train)
>>> y_val = np.log(y_val)
>>> y_test = np.log(y_test)
>>> 
>>> # All the alphas whose RMSE we're going to check
>>> alphas = [0, 0.01, 0.1, 1, 10]
>>> root_mean_squared_errors = dict()
>>> 
>>> # If there are multiple options, select the smallest alpha.
>>> 
>>> 
>>> for alpha in alphas:
>>>     model = Ridge(alpha=alpha, solver="sag", random_state=42)
>>>     x_train = one_hot_encode(df_train)
>>>     model.fit(x_train, y_train)
>>>     x_val = one_hot_encode(df_val)
>>>     y_pred = model.predict(x_val)
>>>     root_mean_squared_errors[alpha] = np.sqrt(mean_squared_error(y_val, y_pred))
>>> 
>>> print({x: round(y, 5) for x, y in root_mean_squared_errors.items()})  # Same RMSEs, thus we select the
""")
# %% Q6 - For this question, we'll see how to use a linear regression model from Scikit-Learn
# We'll need to use the original column 'median_house_value'. Apply the logarithmic transformation to this column.

# Preparing the original validation framework with non-binarized target variable
df_train, df_val, df_test, y_train, y_val, y_test = validation_framework()

# Applying logarithm transformation to the target
y_train = np.log(y_train)
y_val = np.log(y_val)
y_test = np.log(y_test)

# All the alphas whose RMSE we're going to check
alphas = [0, 0.01, 0.1, 1, 10]
root_mean_squared_errors = dict()

# If there are multiple options, select the smallest alpha.


for alpha in alphas:
    model = Ridge(alpha=alpha, solver="sag", random_state=42)
    x_train = one_hot_encode(df_train)
    model.fit(x_train, y_train)
    x_val = one_hot_encode(df_val)
    y_pred = model.predict(x_val)
    root_mean_squared_errors[alpha] = np.sqrt(mean_squared_error(y_val, y_pred))

st.code({x: round(y, 5) for x, y in root_mean_squared_errors.items()})  # Same RMSEs, thus we select the
