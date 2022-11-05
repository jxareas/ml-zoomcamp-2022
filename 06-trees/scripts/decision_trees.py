# %% Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

df = pd.read_feather("../data/credit_scoring.feather")
df.head()

# %% Data Cleaning
df.columns = df.columns.str.lower()

status_values = {
    0: "unknown",
    1: "ok",
    2: "default",
}
df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

for category in ['income', 'assets', 'debt']:
    df[category] = df[category].replace(to_replace=99999999, value=np.nan)

df = df[df.status != "unknown"].reset_index()

del home_values, job_values, marital_values, records_values, status_values, category

# %% Train-Test Validation Split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_validation = train_test_split(df, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_validation = df_validation.reset_index(drop=True)

# Target Variable
y_train = (df_train.status == 'default').astype('int').values
y_validation = (df_validation.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status'], df_validation['status'], df_test['status']


# %% Decision Trees

# Example of a rule in a decision tree
def assess_risk(client):
    if client['records'] == 'yes':
        if client['job'] == 'parttime':
            return 'default'
        else:
            return 'ok'
    else:
        if client['assets'] > 6_000:
            return 'ok'
        else:
            return 'default'


x_i = df_train.iloc[0].to_dict()
assess_risk(x_i)

# %% Transforming the data
train_dicts = df_train.fillna(0).to_dict(orient='records')

dict_vectorizer = DictVectorizer(sparse=False)
X_train = dict_vectorizer.fit_transform(train_dicts)

# Printing the names of each column
dict_vectorizer.get_feature_names_out()

# %% Fitting a Decision Tree Classifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict_proba(X_train)[:, 1]  # Getting the predictions for the positive class
roc_auc_score(y_train, y_pred)  # 1.00 of ROC AUC Score, Overfitting

# %% Testing against the Validation Dataset

validation_dicts = df_validation.fillna(0).to_dict(orient='records')
X_validation = dict_vectorizer.transform(validation_dicts)

y_pred = decision_tree.predict_proba(X_validation)[:, 1]
roc_auc_score(y_validation, y_pred)  # 0.63% of ROC AUC Score

# %% Controlling the depth of the decision tree

decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict_proba(X_train)[:, 1]  # Getting the predictions for the positive class
roc_auc_score(y_train, y_pred)  # 076 of ROC AUC Score, not overfitting anymore

# %% Testing again on the validation data

y_pred = decision_tree.predict_proba(X_validation)[:, 1]
roc_auc_score(y_validation, y_pred)  # 0.74 of ROC AUC Score

# %% Fitting a Decision Stump

decision_stump = DecisionTreeClassifier(max_depth=1)
decision_stump.fit(X_train, y_train)

y_pred = decision_stump.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print("Training AUC: ", auc)  # 0.62 of Training ROC AUC Score

y_pred = decision_stump.predict_proba(X_validation)[:, 1]
auc = roc_auc_score(y_validation, y_pred)
print("Validation AUC:", auc)  # 0.60 of Validation ROC AUC Score

# Extracting the feature names
feature_names = dict_vectorizer.get_feature_names_out().tolist()
rules_text = export_text(decision_stump, feature_names=feature_names)
print(rules_text)

# %% Decision Tree Learning Algorithm

dummy_data = [
    [8000, 'default'],
    [2000, 'default'],
    [0, 'default'],
    [5000, 'ok'],
    [5000, 'ok'],
    [4000, 'ok'],
    [9000, 'ok'],
    [3000, 'default'],
]

dummy_df = pd.DataFrame(dummy_data, columns=['assets', 'status'])
dummy_df.head()

# %% Creating serveral potetion thresholds for a split

thresholds = [2_000, 3_000, 4_000, 5_000, 8_000]
for max_depth in thresholds:
    split_condition = dummy_df.assets > max_depth
    df_right = dummy_df[split_condition]
    df_left = dummy_df[~split_condition]
    print(f"Threshold : {max_depth}")
    print("Right Df")
    print(df_right.status.value_counts(normalize=True).round(2).to_string())
    print("Left Df")
    print(df_left.status.value_counts(normalize=True).round(2).to_string())
    print("-----------------------------------------")

# %% Decision Tree Parameter Tuning
possible_depths = [4, 5, 6, 7, 10, 15, 20, None]
possible_sample_leafs = [1, 2, 5, 10, 15, 20, 100, 200, 500]

tuning_scores = []

# Tuning the Max Depth parameter: size of the tree
for min_leafs in possible_sample_leafs:
    # Tuning the Min Samples Leaf: size of the decision node
    for depth in possible_depths:
        temp_decision_tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leafs)
        temp_decision_tree.fit(X_train, y_train)

        y_prediction = temp_decision_tree.predict_proba(X_validation)[:, 1]
        auc = roc_auc_score(y_validation, y_prediction).__round__(3)
        tuning_scores.append((depth, min_leafs, auc))
        print(f"For depth {depth} & leaf {min_leafs} the auc score is {auc}")

df_scores = pd.DataFrame(tuning_scores, columns=['max_depth', 'min_samples_leaf', 'auc'])
df_scores.sort_values(by='auc', ascending=False, inplace=True)
df_scores.head()

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
sns.heatmap(df_scores_pivot, annot=True)
plt.show()

# %% Fitting a Random Forest

random_forest = RandomForestClassifier(n_estimators=10, random_state=11)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict_proba(X_validation)[:, 1]
random_forest_auc = roc_auc_score(y_validation, y_pred)
print(random_forest_auc)

# %% Iterating over the number of estimators

random_forest_scores = []

for n in range(10, 201, 10):
    random_forest = RandomForestClassifier(n_estimators=n, random_state=11)
    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict_proba(X_validation)[:, 1]
    random_forest_auc = roc_auc_score(y_validation, y_pred)
    random_forest_scores.append((n, random_forest_auc))
    print(f"For number of estimators {n} the ROC AUC score is {random_forest_auc.__round__(2)}")

df_rf_scores = pd.DataFrame(random_forest_scores, columns=["n_estimators", "auc"])
# Top 5 scores
df_rf_scores.sort_values(by="auc", ascending=False).head()

# Plotting the scores
plt.plot(df_rf_scores.n_estimators, df_rf_scores.auc)
plt.show()

# %% Iterating over the number of estimators and max depth parameters
random_forest_scores = []
possible_max_depth = [5, 10, 15]

for max_depth in possible_max_depth:
    for n in range(10, 201, 10):
        random_forest = RandomForestClassifier(n_estimators=n, max_depth=max_depth, random_state=11)
        random_forest.fit(X_train, y_train)

        y_pred = random_forest.predict_proba(X_validation)[:, 1]
        random_forest_auc = roc_auc_score(y_validation, y_pred)
        random_forest_scores.append((max_depth, n, random_forest_auc))
        print(
            f"For number of estimators {n}, max depth {max_depth} the ROC AUC score is {random_forest_auc.__round__(2)}")

df_rf_scores = pd.DataFrame(random_forest_scores, columns=["max_depth", "n_estimators", "auc"])
# Top 5 scores
df_rf_scores.sort_values(by="auc", ascending=False).head()

# Plotting the scores
for max_depth in possible_max_depth:
    df_subset = df_rf_scores[df_rf_scores.max_depth == max_depth]
    plt.plot(df_subset.n_estimators, df_subset.auc, label=max_depth)

plt.title("Random Forest")
plt.xlabel("Number of Estimators")
plt.ylabel("ROC AUC Score")
plt.legend(title="Max Depth")
plt.show()

# %% Iterating over the number of estimators and min sample leafs parameters

selected_max_depth = 10
random_forest_scores = []
possible_sample_leafs = [1, 3, 5, 10, 50]

for s in possible_sample_leafs:
    for n in range(10, 201, 10):
        random_forest = RandomForestClassifier(n_estimators=n,
                                               min_samples_leaf=s,
                                               max_depth=selected_max_depth, random_state=11)
        random_forest.fit(X_train, y_train)

        y_pred = random_forest.predict_proba(X_validation)[:, 1]
        random_forest_auc = roc_auc_score(y_validation, y_pred)
        random_forest_scores.append((s, n, random_forest_auc))
        print(
            f"For number of estimators {n}, min samples leaf {s} the ROC AUC score is {random_forest_auc.__round__(3)}")

df_rf_scores = pd.DataFrame(random_forest_scores, columns=["min_samples_leaf", "n_estimators", "auc"])
# Top 5 scores
df_rf_scores.sort_values(by="auc", ascending=False).head()

# Plotting the scores
for min_samples_leaf in possible_sample_leafs:
    df_subset = df_rf_scores[df_rf_scores.min_samples_leaf == min_samples_leaf]
    plt.plot(df_subset.n_estimators, df_subset.auc, label=min_samples_leaf)

plt.title("Random Forest")
plt.xlabel("Number of Estimators")
plt.ylabel("ROC AUC Score")
plt.legend(title="Min Sample Leafs")
plt.show()

# %% Fitting our final Random Forest Classifier

# Selecting min samples leaf to 3
min_samples_leaf = 3
# Selecting  max depth to 10
max_depth = 10
# Selecting n_estimators to 100
n_estimators = 100

random_forest = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=1,
                                       n_jobs=1)
random_forest.fit(X_train, y_train)

# %% Fitting an Extreme Gradient Boosting Model

d_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
d_validation = xgb.DMatrix(X_validation, label=y_validation, feature_names=feature_names)

xgb_params = {
    'eta': 0.3,  # Learning Rate
    'max_depth': 6,  # Max Depth of a Decision Tree
    'min_child_weight': 1,  # Min Samples Leaf
    'objective': 'binary:logistic',  # Binary Classification - Logistic Model,
    'eval_metric': "auc",  # The Evaluation Metric - ROC AUC Score
    'nthreads': 8,  # Number of Threads
    'seed': 1,  # Random State
    'verbosity': 1,  # Show Only Warnings
}
watchlist = [(d_train, 'train'), (d_validation, 'validation')]
model = xgb.train(xgb_params, d_train,
                  evals=watchlist, verbose_eval=5,
                  num_boost_round=200)

y_pred = model.predict(d_validation)
roc_auc_score(y_validation, y_pred)

# %% Comparing the Models

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict_proba(X_validation)[:, 1]
decision_tree.auc = roc_auc_score(y_validation, y_pred)

# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=1)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict_proba(X_validation)[:, 1]
random_forest.auc = roc_auc_score(y_validation, y_pred)

# XGBoost
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgboost_tree = xgb.train(xgb_params, d_train, num_boost_round=175)
y_pred = model.predict(d_validation)
xgboost_tree.auc = roc_auc_score(y_validation, y_pred)

print(f"Decision Tree AUC {decision_tree.auc}")
print(f"Random Forest AUC {random_forest.auc}")
print(f"XGBoost AUC {xgboost_tree.auc}")
