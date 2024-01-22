import streamlit as st
from streamlit_option_menu import option_menu

from templates.regression import regression
from templates.classification import classification

st.set_page_config(layout="wide")

selected = option_menu(
    menu_title="ML Model Building Application",
    options=["Home", "Regression", "Classification"],
    icons=["house", "graph-up", "diagram-3"],
    orientation="horizontal",
    default_index=0,
)

if selected == "Home":
    with open("./templates/markdown/home.md", "r", encoding="UTF-8") as home_file:
        home_content = home_file.read()

    st.markdown(home_content, unsafe_allow_html=True)

elif selected == "Regression":
    regression()

elif selected == "Classification":
    classification()
