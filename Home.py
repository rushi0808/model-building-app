import streamlit as st


def main():
    st.markdown(
        """
    ## Welcome to Regression and Classification Model Building app! 

    ### Problem Statement:
    - **Building regression and classification models can be time-consuming and complex, requiring coding knowledge.**

    ### Motivation:
    - So based on above problem and writing code every time I wanted to build a model, I decided to build this app with the help of my knowledge in python, machine learning and API like streamlit i was able to achive my target to build this app for building regression and classification models.

    ### Libraries:
    - numpy for mathematical oprations
    - pandas for loading data and preprocessing
    - scikit-learn for model building, preprocessing, model performance report generating
    - matplotlib for building charts in app
    - plotly for building charts in app
    - streamlit for front end
    - ydata profiling for data analysis report


    ### Keyfeature:
    - Algorithm selection: Choose from a variety of regression and classification algorithms to find the best fit for your problem.
    - Hyperparameter tuning: Select and choose the best hyperparameter according to you and see for yourself which is best for you.
    - Real-time model evaluation: Evaluate the performance of your models instantly and make data-driven decisions.
    - Easy to use: app is designed with a user-friendly interface, allowing you to focus on building models rather than dealing with technical details.

    ### Benefits:
    - Save time: This app eliminates the need for manual coding and tedious model building, streamlining the process.

    **Go to regression page or classification page to explore the app!**
    """
    )


if __name__ == "__main__":
    main()
