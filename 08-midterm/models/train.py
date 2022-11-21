import pandas as pd
import numpy as np
import xgboost as xgb
import bentoml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
DATA_PATH = "../data/phone_prices.csv"
df = pd.read_csv(DATA_PATH)

# %% Features and Target
# Removing the `names` and `model` features, which are redundant and unnecesary, only useful to query the data
features = ['brand', 'battery_capacity', 'screen_size',
            'touch_screen', 'resolution_x', 'resolution_y', 'processor', 'ram',
            'internal_storage', 'rear_camera', 'front_camera', 'operating_system',
            'wi_fi', 'bluetooth', 'gps', 'number_of_sims', '3g', '4g_lte']
target = ['price']

# %% Train-Test Split

# Setting a seed (random state)
custom_seed = 287

df_full_train, df_test = train_test_split(df[features + target], test_size=0.2, shuffle=True, random_state=custom_seed)
df_train, df_validation = train_test_split(df_full_train, test_size=0.25, shuffle=True, random_state=custom_seed)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_validation = df_validation.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Applying a Log Transformation to the Target Variable
y_train = np.log1p(df_train[target].values)
y_validation = np.log1p(df_validation[target].values)
y_test = np.log1p(df_test[target].values)

# Removing the target variable
del df_train[target[0]], df_validation[target[0]], df_test[target[0]]

# %% XGBoost Model
dv = DictVectorizer(sparse=False)

dict_train = df_train.to_dict(orient='records')
x_train = dv.fit_transform(dict_train)
features = dv.get_feature_names()
dtrain = xgb.DMatrix(x_train, label=y_train)

dict_val = df_validation.to_dict(orient='records')
x_val = dv.transform(dict_val)
dval = xgb.DMatrix(x_val, label=y_validation)

xgboost_seed = 20
xgb_params = {'seed': xgboost_seed, 'eval_metric': 'rmse', 'n_jobs': -1}
xgboost_model = xgb.train(xgb_params, dtrain)

# %% Saving BentoML Model
bentoml.xgboost.save_model("xgboost_phone_predictor", xgboost_model,
                           custom_objects={
                               "dictVectorizer": dv
                           },
                           signatures={
                               "predict": {
                                   "batchable": True,
                                   "batch_dim": 0,
                               }
                           })
