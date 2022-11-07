# %% Importing Libraries and Dependencies

import re
import subprocess

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

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

# %% Question 1 - Train a Decision Tree Regressor to predict the `median_house_value` variable
# Train a model with `max_depth = 1` (a Decision Stump)

decision_stump = DecisionTreeRegressor(max_depth=1)
decision_stump.fit(X_train, y_train)

# Which feature is used for splitting the data?
print(export_text(decision_tree=decision_stump,
                  feature_names=dict_vectorizer.get_feature_names_out().tolist()))
# `ocean_proximity=INLAND` is the feature used for splitting the data

# %% Question 2 - Train a Random Forest Model with the following parameters:
# n_estimators=10
fixed_n_estimators = 10
# random_state=1
fixed_random_state = 1
# n_jobs=-1 (optional-to make training faster)
fixed_n_jobs = -1

# Training the Random Forest Model
random_forest = RandomForestRegressor(n_estimators=fixed_n_estimators, random_state=fixed_random_state,
                                      n_jobs=fixed_n_jobs)
random_forest.fit(X_train, y_train)
# Calculating Mean Squared Error
y_pred = random_forest.predict(X_validation)
mean_squared_error(y_validation, y_pred)  # 0.06

# %% Question 3 - Experimenting with the `n_estimators` parameter
# Try different values of this parameter from 10 to 200 with step 10.
possible_n_estimators = range(10, 201, 10)
# Set random_state to 1.
scores = []

# Evaluate the model on the validation dataset.
for n_estimator in possible_n_estimators:
    random_forest = RandomForestRegressor(n_estimators=n_estimator, random_state=fixed_random_state,
                                          n_jobs=fixed_n_jobs)
    random_forest.fit(X_train, y_train)
    # Calculating Mean Squared Error
    y_pred = random_forest.predict(X_validation)
    mse = mean_squared_error(y_validation, y_pred)
    scores.append((n_estimator, mse))

df_scores = pd.DataFrame(data=scores, columns=['n_estimator', 'mse'])

# Plotting the `n_estimator` parameter against the `mean squared error`
plt.plot(df_scores.n_estimator, df_scores.mse)
plt.show()

# After approx. `n_estimator=70` the RMSE stops improving

# %% Question 4 - Let`s select the best `max_depth`
# Try different values of max_depth: [10, 15, 20, 25] For each of these values
# Try different values of n_estimators from 10 till 200 (with step 10)
# Fix the random seed: random_state=1
# What's the best max depth?
trained_models = pd.read_feather('../data/trained_models.feather')
trained_models['rmse'] = np.sqrt(trained_models.mse)

for depth in [10, 15, 20, 25]:
    df_subset = trained_models[trained_models.max_depth == depth]
    plt.plot(df_subset.n_estimators, df_subset.rmse,
             label=depth)

plt.title('Number of Estimators vs Root Mean Square Error')
plt.xlabel('Number of Estimators')
plt.ylabel('Root Mean Square Error')
plt.legend(title='Max Depth')
plt.show()  # As evidenced by thje graph, the best max depth is 25

# %% Question 5 - Feature Importance

# At each step of the decision tree learning algorith, it finds the best split.
# When doing it, we can calculate "gain" - the reduction in impurity before and after the split.
# This gain is quite useful in understanding what are the imporatnt features for tree-based models.
#
# In Scikit-Learn, tree-based models contain this information in the feature_importances_ field.
#
# For this homework question, we'll find the most important feature:
#
# Train the model with these parametes:
# n_estimators=10,
# max_depth=20,
# random_state=1,
# n_jobs=-1 (optional)
random_forest = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=fixed_random_state, n_jobs=-1)
random_forest.fit(X_train, y_train)
# Get the feature importance information from this model
feature_importance = random_forest.feature_importances_.tolist()
feature_names = dict_vectorizer.get_feature_names_out().tolist()
feature_df = pd.DataFrame({
    'feature_names': feature_names,
    'feature_importance': feature_importance,
})

feature_df.sort_values(by='feature_importance', ascending=False)  # Median Income is the most important feature

# %% Question 6 - XGBoosting
# Now let's train an XGBoost model! For this question, we'll tune the eta parameter:
#
# Install XGBoost
# Create DMatrix for train and validation
# Create a watchlist
# Train a model with these parameters for 100 rounds:
# @ CREATING THE DMARTIX:
features = dict_vectorizer.feature_names_

regex = re.compile(r"<", re.IGNORECASE)
features = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in features]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_validation, label=y_validation, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]
scores = {}


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))

    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results


xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model1 = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5, evals=watchlist)
xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model2 = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5, evals=watchlist)

# Which eta leads to the best RMSE score on the validation dataset?
y_pred_1 = model1.predict(dval)
mse1 = mean_squared_error(y_validation, y_pred_1)
print(f"MSE1: {mse1}")

y_pred_2 = model2.predict(dval)
mse2 = mean_squared_error(y_validation, y_pred_2)
print(f"MSE2: {mse2}")
# Both
