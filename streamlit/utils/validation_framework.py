import numpy as np
import pandas as pd

features = ['latitude', 'longitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = ['median_house_value_log']


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
