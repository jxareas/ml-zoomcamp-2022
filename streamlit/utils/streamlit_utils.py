import streamlit as st


def top_header(question, title):
    st.subheader(question)
    st.subheader(title)


def separator():
    return st.write("---")
