# %% Importing Libraries and Dependencies

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

target = 'median_house_value'
selected_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                    "households", "median_income", target, "ocean_proximity"]
df = pd.read_feather("../data/housing.feather")[selected_columns]
df.head()

# %% Data Cleaning and Preparation

# Checking for na's
df.isna().any()  # Total Bedrooms column has NAs

df['total_bedrooms'] = df.total_bedrooms.fillna(0)

# Applying the Log Transformation to the Target Variable
df[target] = np.log(df[target])

# %% Train-Test Split

random_state = 1
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
df_train, df_validation = train_test_split(df_full_train, test_size=0.25, random_state=random_state)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_validation = df_validation.reset_index(drop=True)

y_train = df_train[target].values
y_validation = df_validation[target].values
y_test = df_test[target].values

del df_train[target], df_test[target], df_validation[target]

# %% Turning DataFrames into Matrices with DictVectorizer

dict_vectorizer = DictVectorizer(sparse=False)

train_dict = df_train.fillna(0).to_dict(orient='records')
X_train = dict_vectorizer.fit_transform(train_dict)

validation_dict = df_validation.fillna(0).to_dict(orient='records')
X_validation = dict_vectorizer.transform(validation_dict)


