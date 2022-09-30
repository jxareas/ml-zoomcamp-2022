import streamlit as st


def top_header(title, subtitle):
    st.subheader(title)
    st.subheader(subtitle)


def top_question(question_number, title):
    question = f"Question {question_number}:"
    st.subheader(question)
    st.subheader(title)


def br():
    return st.write("---")
