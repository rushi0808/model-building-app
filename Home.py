import streamlit as st


def main():
    st.markdown(
        """
    # Regression and Classification model building App!

    ### About Project

    - This is a app for building regression and classification models.
    - Build with the help of python and streamlit api.
    - Where you can upload your csv or excel file.
    - Choose the page for the model you want to build regression or classification.
    - Choose the model us want to build or explore all the models and see for your self which suites your dataset the most.
    - Requirements included.

    ### Features

    - Included variety of models for classification and regression.
    - Included hyperparameter tuning so user can choose for him self the best parameters for data.
    - Included model performance report for train and test data.
    - Also added pandas profiling for data analysis.
    - Also added data transformation feature for those who want to do standerScaling, minmax scaling and log transformation.
    - Downloading model with scaler for future predictions.
    """
    )


if __name__ == "__main__":
    main()
