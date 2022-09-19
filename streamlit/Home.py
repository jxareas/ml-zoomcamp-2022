import streamlit as st
import pandas as pd
from utils import lottie
from streamlit_lottie import st_lottie

lottie_robot_animation = lottie.load_url("https://assets9.lottiefiles.com/packages/lf20_xaxycw1s.json")

st.set_page_config(
    page_title="Zoomcamp Solutions",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("Zoomcamp Solutions 2022")

st.header("Welcome to Zoomcamp Solutions!")

st.markdown(
    "This simple Streamlit app provides a simple navigation through [my solutions](https://github.com/jxareas/machine-learning-bookcamp-2022) for the"
    " **Machine Learning Zoomcamp 2022.** Feel free to learn or point out any issues.")

st_lottie(lottie_robot_animation, height=200, quality="high")
