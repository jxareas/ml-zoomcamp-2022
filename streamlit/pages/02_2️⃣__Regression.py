import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from utils.streamlit_utils import top_question, br, top_header
from utils import validation_framework as val

st.set_page_config(layout="wide")

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

st.title("Homework 2: Machine Learning for Regression")
st.markdown("""
This is a chapter that explores the theme of Regression in Machine Learning.
All algorithms in this chapter are implemented manually, using Linear Algebra knowledge.

Here are some of the topics covered by this chapter:

* Doing **Exploratory Data Analysis** to discover patterns
* Using **Linear Regression** via the Normal Equation, to predict quantitative target variables.
* Applying **Tikhonov Regularization** to a Linear Regression model.
* Using **random seeds** in order to control reproducibility
""")
br()
st.subheader("Data Preview: ")

df = pd.read_csv(url)
st.dataframe(df)
br()

# %% Exploratory Data Analysis
top_header("Exploratory Data Analysis", "Does the `median_house_value` variable have a long tail?")
histogram_figure = px.histogram(data_frame=df, x="median_house_value", template="ggplot2")
st.plotly_chart(histogram_figure)
st.markdown("We can appreciate the distribution of the `median_house_value` is skewed to the right.")
br()

# %% Feature Selection

st.subheader("Feature Selection")
st.write("Selecting only the relevant features for this homework.")

st.code("""
>>> # For the rest of the homework, you'll need to use only these columns
>>> features = ['latitude', 'longitude', 'housing_median_age', 'total_rooms',
>>>            'total_bedrooms', 'population', 'households', 'median_income']
>>> target = ['median_house_value_log']
>>> df['median_house_value_log'] = np.log1p(df.median_house_value)
>>> df.head()
""")

columns = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
           'median_income', 'median_house_value']
df['median_house_value_log'] = np.log1p(df.median_house_value)
st.write(df[columns].head(5))
br()

# %% Question 1
top_question(1, "Find a feature with missing values. How many missing values does it have?")

st.code("""
>>> na_per_column = df.isna().sum()
>>> na_per_column[na_per_column > 0]
""")
st.code(df.isna().sum(axis=0))
br()

# %% Question 2
top_question(2, "What's the median (50% percentile) for variable population?")

st.code("""
>>> population_median = np.median(df.population)
>>> population_median
""")
st.code(np.median(df.population))
br()

# %% Prepare the Validation Framework
st.subheader("Preparing the Validation Framework")
st.write("Defining functions to prepare the validation framework.")
st.code("""
def shuffle(data, random_seed):
    n = len(data)
    np.random.seed(random_seed)
    index = np.arange(n)
    np.random.shuffle(index)

    shuffled_data = data.iloc[index].reset_index(drop=True)
    return shuffled_data


def split(data, validation_percent=0.2, test_percent=0.2):
    n = len(data)
    n_val = int(n * validation_percent)
    n_test = int(n * test_percent)
    n_train = n - n_val - n_test

    df_train = data.iloc[:n_train]
    df_val = data.iloc[n_train:n_train + n_val]
    df_test = data.iloc[n_train + n_val:]

    return df_train, df_val, df_test


def prepare_validation_framework(data, random_seed=42):
    data = shuffle(data.copy(), random_seed)

    n = len(data)
    n_val = int(n * 0.2)  # Validation size
    n_test = int(n * 0.2)  # Test size
    n_train = n - n_val - n_test  # Train size

    # Performing the train-test split
    df_train = data.iloc[:n_train][features]
    df_val = data.iloc[n_train:n_train + n_val][features]
    df_test = data.iloc[n_train + n_val:][features]

    target_train = data.iloc[:n_train][target].squeeze()
    target_val = data.iloc[n_train:n_train + n_val][target].squeeze()
    target_test = data.iloc[n_train + n_val:][target].squeeze()

    return df_train, df_val, df_test, target_train, target_val, target_test


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    # Regularization
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full


def train_linear_regression(X, y):
    ones = np.ones(len(X))
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    betas = XTX_inv.dot(X.T).dot(y)

    return betas


def predict(X, betas):
    return betas[0] + X.dot(betas[1:])


def rmse(y, y_hat):
    squared_errors = (y - y_hat) ** 2
    mean_squared_error = np.mean(squared_errors)
    return np.sqrt(mean_squared_error)
""")
br()

# %% Question 3

top_question(3, "Which options give the best *RMSE*?")
st.markdown("""
We need to deal with missing values for the column from Q1.

We have two options: fill it with 0 or with the mean of this variable.

Try both options. For each, train a linear regression model without regularization using the code from the lessons.

For computing the mean, use the training only!

Use the validation dataset to evaluate the models and compare the RMSE of each option.

Round the RMSE scores to 2 decimal digits using round(score, 2)

Which option gives better RMSE?
""")

st.subheader("Approach 1 : Filling with Zeros")
st.code("""
x_train, x_val, x_test, y_train, y_val, y_test = prepare_validation_framework(df, random_seed=42)

zero_linear_model = train_linear_regression(x_train.fillna(0).values, y_train.values)
zero_predicted_y_val = predict(x_val, zero_linear_model)
zero_rmse = rmse(y_val.values, zero_predicted_y_val)
zero_rmse
""")
x_train, x_val, x_test, y_train, y_val, y_test = val.prepare_validation_framework(df, random_seed=42)

zero_linear_model = val.train_linear_regression(x_train.fillna(0).values, y_train.values)
zero_predicted_y_val = val.predict(x_val, zero_linear_model)
zero_rmse = val.rmse(y_val.values, zero_predicted_y_val)
st.code(zero_rmse)

st.subheader("Approach 2: Filling with NAs")
st.code("""
total_bedrooms_mean = x_train.total_bedrooms.mean()
mean_linear_model = train_linear_regression(x_train.fillna(total_bedrooms_mean).values, y_train.values)
mean_predicted_y_val = predict(x_val, mean_linear_model)
mean_rmse = rmse(y_val.values, mean_predicted_y_val)
mean_rmse
""")
total_bedrooms_mean = x_train.total_bedrooms.mean()
mean_linear_model = val.train_linear_regression(x_train.fillna(total_bedrooms_mean).values, y_train.values)
mean_predicted_y_val = val.predict(x_val, mean_linear_model)
mean_rmse = val.rmse(y_val.values, mean_predicted_y_val)
st.code(mean_rmse)

st.write(
    "Hence, we can safely conclude that both approaches are equally good, as the difference between both RMSEs is almost null.")

# %% Question 4
top_question(4, "Select the `r` with the smallest *RMSE*")
st.markdown("""
Now let's train a regularized linear regression.

For this question, fill the NAs with 0.

Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].

Use RMSE to evaluate the model on the validation dataset.

Round the RMSE scores to 2 decimal digits.

Which r gives the best *RMSE*?

If there are multiple options, select the smallest `r`.
""")
st.code("""
>>> multipliers = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
>>> rmse_per_multiplier = dict()
>>> for multiplier in multipliers:
>>>    model = train_linear_regression_reg(x_train.fillna(0).values, y_train.values, r=multiplier)
>>>    model_prediction_val = predict(x_val, model)
>>>    model_rmse = rmse(y_val.values, model_prediction_val)
>>>    rmse_per_multiplier[multiplier] = model_rmse
>>>    
>>>    pp({key : round(value, 2) for key, value in rmse_per_multiplier.items()})
""")
multipliers = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
rmse_per_multiplier = dict()
for multiplier in multipliers:
    model = val.train_linear_regression_reg(x_train.fillna(0).values, y_train.values, r=multiplier)
    model_prediction_val = val.predict(x_val, model)
    model_rmse = val.rmse(y_val.values, model_prediction_val)
    rmse_per_multiplier[multiplier] = model_rmse

st.code({key: round(value, 2) for key, value in rmse_per_multiplier.items()})
br()

# %% Question 5
top_question(5, "Compute the standard deviation of RMSE scores")
st.markdown("""
We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.

Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

For each seed, do the train/validation/test split with 60%/20%/20% distribution.

Fill the missing values with 0 and train a model without regularization.

For each seed, evaluate the model on the validation dataset and collect the RMSE scores.

What's the standard deviation of all the scores? To compute the standard deviation, use np.std.

Round the result to 3 decimal digits using round(std, 3).
""")
st.code("""
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_per_seed = dict()
for seed in seeds:
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_validation_framework(df, random_seed=seed)
    model = train_linear_regression(x_train.fillna(0).values, y_train.values)
    model_prediction_val = predict(x_val, model)
    model_rmse = rmse(y_val.values, model_prediction_val)
    rmse_per_seed[seed] = model_rmse
deviation = np.std([value for value in rmse_per_seed.values()])
print(round(deviation, 3))
""")
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_per_seed = dict()
for seed in seeds:
    x_train, x_val, x_test, y_train, y_val, y_test = val.prepare_validation_framework(df, random_seed=seed)
    model = val.train_linear_regression(x_train.fillna(0).values, y_train.values)
    model_prediction_val = val.predict(x_val, model)
    model_rmse = val.rmse(y_val.values, model_prediction_val)
    rmse_per_seed[seed] = model_rmse
deviation = np.std([value for value in rmse_per_seed.values()])
st.code(round(deviation, 3))
br()

# %% Question 6
top_question(6, "Compute the Test RMSE")
st.markdown("""
Split the dataset like previously, use seed 9.

Combine train and validation datasets.

Fill the missing values with 0 and train a model with r=0.001.

What's the RMSE on the test dataset?
""")
st.code("""
>>> x_train, x_val, x_test, y_train, y_val, y_test = prepare_validation_framework(df, random_seed=9)
>>> full_x_train_set = pd.concat([x_train, x_val]).fillna(0)
>>> full_y_train_set = pd.concat([y_train, y_val]).fillna(0)
>>> 
>>> reg_model = train_linear_regression_reg(full_x_train_set.values, full_y_train_set.values, r=0.001)
>>> reg_model_test_pred = predict(x_test, reg_model)
>>> reg_model_rmse = rmse(y_test, reg_model_test_pred)
>>> print(reg_model_rmse)
""")
x_train, x_val, x_test, y_train, y_val, y_test = val.prepare_validation_framework(df, random_seed=9)
full_x_train_set = pd.concat([x_train, x_val]).fillna(0)
full_y_train_set = pd.concat([y_train, y_val]).fillna(0)

reg_model = val.train_linear_regression_reg(full_x_train_set.values, full_y_train_set.values, r=0.001)
reg_model_test_pred = val.predict(x_test, reg_model)
reg_model_rmse = val.rmse(y_test, reg_model_test_pred)
st.code(reg_model_rmse)
