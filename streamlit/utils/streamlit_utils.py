import streamlit as st
import io
import requests
import pandas as pd
import numpy as np

@st.cache(allow_output_mutation=True)
def get_dataframe(data_url):
    r = requests.get(data_url)
    df = pd.read_csv(io.StringIO(r.text), encoding='utf8', sep=",", dtype={"switch": np.int8})
    return df

def top_header(title, subtitle):
    st.subheader(title)
    st.subheader(subtitle)


def top_question(question_number, title):
    question = f"Question {question_number}:"
    st.subheader(question)
    st.subheader(title)


def br():
    return st.write("---")
