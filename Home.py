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

<<<<<<< HEAD
    - This is a app for building regression and classification models.
    - Build with the help of python and streamlit api.
    - Where you can upload your csv or excel file.
    - Choose the page for the model you want to build regression or classification.
    - Choose the model us want to build or explore all the models and see for your self which suites your dataset the most.
    - Requirements included.
=======
if selected == "Home":
    with open("./templates/markdown/home.md", "r", encoding="UTF-8") as home_file:
        home_content = home_file.read()
>>>>>>> dev_home_page

    st.markdown(home_content, unsafe_allow_html=True)

<<<<<<< HEAD
    - Included variety of models for classification and regression.
    - Included hyperparameter tuning so user can choose for him self the best parameters for data.
    - Included model performance report for train and test data.
    - Also added pandas profiling for data analysis.
    - Also added data transformation feature for those who want to do standerScaling, minmax scaling and log transformation.
    - Downloading model with scaler for future predictions.
    """
    )
=======
elif selected == "Regression":
    regression()
>>>>>>> dev_home_page

elif selected == "Classification":
    classification()
