import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from datetime import date


today = date.today()
st.sidebar.image("/Users/xiyuanzhang/Desktop/Iwant.png", use_column_width=True)
page = st.sidebar.selectbox("Explore Or Predict", ("Explore","Predict"))
st.sidebar.text(f"Edited by Xiyuan\nModifyied on {today.strftime('%b %d, %Y')}")
if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
