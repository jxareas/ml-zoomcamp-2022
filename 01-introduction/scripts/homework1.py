# %% Importing libraries

import pandas as pd
import numpy as np

df = pd.read_feather("../data/cars.feather")

# %% Q1 -What's the version of numpy you have installed?

print(np.__version__)  # 1.21.5

# %% Q2 - How many records are in the dataset?

len(df)  # 11'914

# %% Q3 - Who are the most frequent car manufacturers (top 3) according
# to the dataset?

df.Make.value_counts().nlargest(3)  # Chevrolet, Ford, Volkswagen

# %% Q4 - What's the number of unique Audi car models in the dataset?

df[df.Make == 'Audi'].Model.nunique()  # 34

# %% Q5 - How many columns in the dataset have missing values?

missing_values_per_column = df.isna().sum()
len(missing_values_per_column[missing_values_per_column > 0])  # 5

# %% Q6:
# Find the median value of "Engine Cylinders" column in the dataset.
cylinders = df['Engine Cylinders']
initial_median = np.median(cylinders[cylinders.notna()])  # 6
# Next, calculate the most frequent value of the same "Engine Cylinders".
most_frequent = cylinders.mode()[0]  # 4
# Use the `fillna` method to fill the missing values in "Engine Cylinders"
# with the most frequent value from the previous step.
cylinders.fillna(value=most_frequent, inplace=True)
# Now, calculate the median value of "Engine Cylinders" once again.
final_median = np.median(cylinders)
# Does the median value change after filling missing values?
print(initial_median != final_median)  # False, both medians are the same

# %% Q7:
# 7.1) Select all the "Lotus" cars from the dataset.
lotus = df[df.Make == 'Lotus']
# Select only columns "Engine HP", "Engine Cylinders".
lotus = lotus[['Engine HP', 'Engine Cylinders']]
# Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
lotus = lotus.drop_duplicates()
# Get the underlying NumPy array. Let's call it X.
X = lotus.to_numpy()
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T.
# Let's call the result XTX.
XTX = np.matmul(X.T, X)
# Invert XTX.
XTX_inv = np.linalg.inv(XTX)
# Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = XTX_inv.dot(X.T).dot(y)
# What's the value of the first element of w?
print(w[0])  # 4.59
