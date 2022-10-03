import streamlit as st
from utils import lottie
from streamlit_lottie import st_lottie

lottie_robot_animation = lottie.load_url("https://assets9.lottiefiles.com/packages/lf20_xaxycw1s.json")

st.set_page_config(
    page_title="Zoomcamp Solutions",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("Zoomcamp Solutions")

st.header("Welcome!")

st.markdown(
    "This simple Streamlit app provides a simple navigation through [my solutions](https://github.com/jxareas/machine-learning-bookcamp-2022) for the"
    " **Machine Learning Zoomcamp of 2022**, as I learn the Python programming language as well as Machine Learning.")
st.markdown("Feel free to check the code, learn or point out any issues!")

st_lottie(lottie_robot_animation, height=200, quality="high")
