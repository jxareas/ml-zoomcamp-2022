# %% Importing Libraries and loading the data

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %% Exploratory Data Analysis

# Load the data
df = pd.read_feather("../data/housing.feather")

# Look at the median house value. Does it have a long tail?
df.median_house_value.hist()
plt.show()  # Histogram that's skewed to the right

# For the rest of the homework, you'll need to use only these columns
features = ['latitude', 'longitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = ['median_house_value_log']
df['median_house_value_log'] = np.log1p(df.median_house_value)

# %% Q1 - Find a feature with missing values. How many missing values does it have?

na_per_column = df.isna().sum()
print(na_per_column[na_per_column > 0])

# %% Question 2 - What's the median (50% percentile) for variable 'population'?

population_median = np.median(df.population)
print(population_median)


# %% Split the data
# This code chunk contains the necessary functions to shuffle the dataframe,
# train the linear regression model & do the train-test split.

def shuffle(data, random_seed):
    """ Returns a shuffled version of the original dataframe
    :param pd.DataFrame data: a dataframe
    :param float random_seed: a seed that controls reproducibility
    :return: Shuffled data
    :rtype: pd.DataFrame
    """
    n = len(data)
    np.random.seed(random_seed)
    index = np.arange(n)
    np.random.shuffle(index)

    shuffled_data = data.iloc[index].reset_index(drop=True)
    return shuffled_data


def split(data, validation_percent=0.2, test_percent=0.2):
    """
    :param pd.DataFrame data: a dataframe
    :param float validation_percent: validation set percentage
    :param float test_percent: test set percentage
    :return: the train, validation and test set
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    n = len(data)
    n_val = int(n * validation_percent)
    n_test = int(n * test_percent)
    n_train = n - n_val - n_test

    df_train = data.iloc[:n_train]
    df_val = data.iloc[n_train:n_train + n_val]
    df_test = data.iloc[n_train + n_val:]

    return df_train, df_val, df_test


def prepare_validation_framework(data, random_seed=42):
    """ Prepares the validation framework for a machine learning model
    :param pd.DataFrame data: a dataframe
    :param float random_seed: a seed that controls reproducibility
    :return: the train, validation and test set, separated by features (X) and target (y)
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
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
    """ Estimates the parameters of a regularized linear regression model
    :param np.ndarray X: a matrix
    :param np.ndarray y: a vector
    :param np.ndarray r: the Lagrange multiplier
    :return: the beta coefficients of the model
    :rtype: np.ndarray
    """
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    # Regularization
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full


def train_linear_regression(X, y):
    """ Estimates the parameters of a linear regression model
    :param X: a matrix
    :param y: a vector
    :return: the beta coefficients of the model
    :rtype: np.ndarray
    """
    ones = np.ones(len(X))
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    betas = XTX_inv.dot(X.T).dot(y)

    return betas


def predict(X, betas):
    """ Predicts values using a linear regression model
    :param np.ndarray X: Design matrix
    :param np.ndarray betas: Beta coefficients
    :return: Estimated target values
    :rtype: np.ndarray
    """
    return betas[0] + X.dot(betas[1:])


def rmse(y, y_hat):
    """ Computes the Root Mean Square Error
    :param y: vector
    :param y_hat: estimated vector
    :return: the root-mean-square error
    :rtype: np.ndarray
    """
    squared_errors = (y - y_hat) ** 2
    mean_squared_error = np.mean(squared_errors)
    return np.sqrt(mean_squared_error)


# %% Q3 - We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
# Try both options. For each, train a linear regression model without regularization using the code from the lessons.
# For computing the mean, use the training only!
# Use the validation dataset to evaluate the models and compare the RMSE of each option.
# Round the RMSE scores to 2 decimal digits using round(score, 2)
# Which option gives better RMSE?

# %% Approach 1 : Fill with 0

# Preparing the train-test split
x_train, x_val, x_test, y_train, y_val, y_test = prepare_validation_framework(df, random_seed=42)

zero_linear_model = train_linear_regression(x_train.fillna(0).values, y_train.values)
zero_predicted_y_val = predict(x_val, zero_linear_model)
zero_rmse = rmse(y_val.values, zero_predicted_y_val)
print(zero_rmse)

# %% Approach 2 : Fill with mean
total_bedrooms_mean = x_train.total_bedrooms.mean()
mean_linear_model = train_linear_regression(x_train.fillna(total_bedrooms_mean).values, y_train.values)
mean_predicted_y_val = predict(x_val, mean_linear_model)
mean_rmse = rmse(y_val.values, mean_predicted_y_val)
print(mean_rmse)

# %% Q4 - Now let's train a regularized linear regression.
# For this question, fill the NAs with 0.
# Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
# Use RMSE to evaluate the model on the validation dataset.
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
multipliers = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
rmse_per_multiplier = dict()
for multiplier in multipliers:
    model = train_linear_regression_reg(x_train.fillna(0).values, y_train.values, r=multiplier)
    model_prediction_val = predict(x_val, model)
    model_rmse = rmse(y_val.values, model_prediction_val)
    rmse_per_multiplier[multiplier] = model_rmse
print(rmse_per_multiplier)

# %% Q5 - We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# Fill the missing values with 0 and train a model without regularization.
# For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
# What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3))
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_per_seed = dict()
for seed in seeds:
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_validation_framework(df, random_seed=seed)
    model = train_linear_regression(x_train.fillna(0).values, y_train.values)
    model_prediction_val = predict(x_val, model)
    model_rmse = rmse(y_val.values, model_prediction_val)
    rmse_per_seed[seed] = model_rmse
deviation = np.std([value for value in rmse_per_seed.values()])
print(deviation)

# %% Q6 - Split the dataset like previously, use seed 9.
# Combine train and validation datasets.
# Fill the missing values with 0 and train a model with r=0.001.
# What's the RMSE on the test dataset?
x_train, x_val, x_test, y_train, y_val, y_test = prepare_validation_framework(df, random_seed=9)
full_x_train_set = pd.concat([x_train, x_val]).fillna(0)
full_y_train_set = pd.concat([y_train, y_val]).fillna(0)

reg_model = train_linear_regression_reg(full_x_train_set.values, full_y_train_set.values, r=0.001)
reg_model_test_pred = predict(x_test, reg_model)
reg_model_rmse = rmse(y_test, reg_model_test_pred)
print(reg_model_rmse)
