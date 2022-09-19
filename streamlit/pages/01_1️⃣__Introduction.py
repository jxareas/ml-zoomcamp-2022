import streamlit as st
import subprocess
import io
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"


@st.cache
def get_dataframe(data_url):
    data = pd.read_csv(url)
    return data


st.title("Homework 1: Introduction to Machine Learning")
st.write(
    "This homework is mostly about basic data wrangling with Pandas & Numpy, as well as some elementary Linear Algebra operations, like matrix multiplication, transpose and the dot product.")
st.write("---")
st.subheader("Data Preview: ")

df = get_dataframe(url)
st.dataframe(df)

# %% Question 1
st.subheader("Q1: What's the version of numpy you have installed?")

st.code(">>> np.__version__")
st.code(np.__version__)

# %% Question 2
st.subheader("Q2: How many records are in the dataset?")

st.code(">>> len(df)")
st.code(len(df))

# %% Question 3
st.subheader("Q3: Who are the most frequent car manufacturers (top 3) according")

st.code(">>> df['Make'].value_counts().nlargest(3)")
st.code(df['Make'].value_counts().nlargest(3))

# %% Question 4
st.subheader("Q4: What's the number of unique Audi car models in the dataset?")

st.code(">>> df[df.Make == 'Audi'].Model.nunique()")
st.code(df[df.Make == 'Audi'].Model.nunique())

# %% Question 5
st.subheader("Q5: How many columns in the dataset have missing values?")

st.code("""
>>> missing_values_per_column = df.isna().sum()
>>> len(missing_values_per_column[missing_values_per_column > 0])  
""")
st.code(sum(df.isna().sum(axis=0) > 0))

# %% Question 6
st.subheader("Q6: Does the median value change after filling missing values?")

median_before = df["Engine Cylinders"].median()
most_frequent_value = df["Engine Cylinders"].mode()
df["Engine Cylinders"].fillna(most_frequent_value, inplace=True)

median_after = df["Engine Cylinders"].median()

st.code("""
# Find the median value of "Engine Cylinders" column in the dataset.
>>> cylinders = df['Engine Cylinders']
>>> initial_median = np.median(cylinders[cylinders.notna()])  # 6
# Next, calculate the most frequent value of the same "Engine Cylinders".
>>> most_frequent = cylinders.mode()[0]  # 4
# Use the `fillna` method to fill the missing values in "Engine Cylinders"
# with the most frequent value from the previous step.
>>> cylinders.fillna(value=most_frequent, inplace=True)
# Now, calculate the median value of "Engine Cylinders" once again.
>>> final_median = np.median(cylinders)
# Does the median value change after filling missing values?
>>> print(initial_median != final_median)  # False, both medians are the same
""")
st.code(median_before != median_after)

# %% Question 7
st.subheader("Q7: Value of the first element of w")

lotus_df = df.loc[df['Make'] == "Lotus", ["Engine HP", "Engine Cylinders"]]
lotus_without_duplicates = lotus_df.drop_duplicates()
X = lotus_without_duplicates.values
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
w = XTX_inv.dot(X.T).dot(y)

st.code("""
# 7.1) Select all the "Lotus" cars from the dataset.
>>> lotus = df[df.Make == 'Lotus']
# Select only columns "Engine HP", "Engine Cylinders".
>>> lotus = lotus[['Engine HP', 'Engine Cylinders']]
# Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
>>> lotus = lotus.drop_duplicates()
# Get the underlying NumPy array. Let's call it X.
>>> X = lotus.to_numpy()
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T.
# Let's call the result XTX.
>>> XTX = np.matmul(X.T, X)
# Invert XTX.
>>> XTX_inv = np.linalg.inv(XTX)
# Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
>>> y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
>>> w = XTX_inv.dot(X.T).dot(y)
# What's the value of the first element of w?
>>> print(w[0])  # 4.59
""")

st.code(w[0])
