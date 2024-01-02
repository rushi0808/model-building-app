import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from streamlit_pandas_profiling import st_profile_report

from src.functions import (
    build_model,
    chi,
    col_drop_df,
    corr,
    data_clean,
    data_smapling,
    data_transform,
    datasummary,
    del_file,
    label_encode,
    load_data,
    model_zip,
    profiling_rp,
    rec_feat_ele,
    save_model,
)


def model_button():
    st.session_state.show_res = not st.session_state.show_res


def model_result(model_ele, x_labels, target, params):
    st.title(f"{model_ele} results:")
    try:
        model = build_model(model_ele, params)
        tr_x, tr_y, ts_x, ts_y = data_smapling(x_labels, target)

        model.fit(tr_x, tr_y)
        st.write(100 * "-")
        st.subheader(f"Train Data Results:")
        pred_tr = model.predict(tr_x)
        st.markdown(f"**Confusion matrix:**")
        st.dataframe(confusion_matrix(tr_y, pred_tr), hide_index=False)
        st.markdown("**Report:**")
        report = classification_report(tr_y, pred_tr, output_dict=True)
        st.dataframe(report)
        st.write(100 * "-")
        st.subheader(f"Test Data Results:")
        pred_ts = model.predict(ts_x)
        st.markdown("**Confusion matrix:**")
        st.dataframe(confusion_matrix(ts_y, pred_ts), hide_index=False)
        st.markdown("**Report:**")
        report = classification_report(ts_y, pred_ts, output_dict=True)
        st.dataframe(report)
        pred_prob = model.predict_proba(ts_x)
        num_class = model.classes_
        for i in num_class:
            l = [1 if x == i else 0 for x in ts_y]
            fpr, tpr, thresh = roc_curve(l, pred_prob[:, i])
            roc_scr = roc_auc_score(l, pred_prob[:, i])
            fig, ax = plt.subplots()
            ax = plt.plot(fpr, tpr, color="r")
            plt.xlabel(f"Class {i} fpr")
            plt.ylabel(f"Class {i} tpr")
            plt.title(f"ROC Curve class {i} vs rest.")
            plt.text(x=0.3, y=0.3, s=f"Area under curve: {round(roc_scr,2)}.")
            st.write(100 * "-")
            st.pyplot(fig)
        st.sidebar.success("Model build Sucessfully!")

        save_model(model)
        model_zip()

        st.sidebar.download_button(
            label="Download model",
            data=open(r"./model_zip.zip", "rb").read(),
            file_name="model_zip.zip",
        )
        del_file()
        # st.sidebar.success("Model downloaded!")

    except Exception as e:
        st.warning(e)


def main():
    st.title("Classification app")
    file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"], help="Upload CSV or Excel file.")
    if file:
        data = load_data(file)
        data_option = st.sidebar.radio(label="Go to: ", options=["Dataframe", "Profile Report"])
        if data.shape[0] < 30:
            st.warning("Model building with less than 30 records is not supported!")

        cate_col = data.select_dtypes("object").columns
        num_col = data.select_dtypes(np.number).columns

        if data_option == "Dataframe":
            st.title("Dataframe: ")
            st.dataframe(data)
        elif data_option == "Profile Report":
            st.title("Profile Report: ")
            pr = profiling_rp(data)
            st_profile_report(pr)

        st.subheader(f"Dataset Summary: ")
        summary, rows, cols = datasummary(data)
        st.dataframe(summary)
        st.write(f"No of rows: {rows}.")
        st.write(f"No of columns: {cols}.")

        col_drop = st.sidebar.multiselect(
            label="Columns you wish to drop.",
            options=sorted(data.columns),
            help="Select columns which have null percentage more then 40% and columns which are not significant.",
        )
        if col_drop:
            sel_target = st.sidebar.selectbox(
                label="Select target Variable",
                options=[col for col in data.columns if col not in col_drop],
                help="Select varable that ypu want to predict make sure that the variable is continous.",
            )
            df = col_drop_df(data, col_drop)
            clean_df = data_clean(df)
            label_encode_df = label_encode(clean_df)

        else:
            sel_target = st.sidebar.selectbox(
                label="Select target Variable",
                options=data.columns,
                help="Select varable that you want to predict make sure that the variable is continous.",
            )
            df = data
            clean_df = data_clean(df)
            label_encode_df = label_encode(clean_df)
        try:
            col_to_trans = st.sidebar.multiselect(
                label="Select columns to transform.",
                options=[col for col in num_col if col not in col_drop],
                help="Select the columns on which you want to apply StandarScaling, min max scaling or log tranformation.",
            )
            trans_method = st.sidebar.radio(
                label="Select transform method:",
                options=[
                    "None",
                    "StandarScale",
                    "Min Max Scaler",
                    "Log transformation",
                ],
                help="Select the method of data transformation as best fit for your data.",
            )

            if trans_method != "None":
                label_encode_df = data_transform(trans_method, label_encode_df, col_to_trans)
        except Exception as e:
            st.warning(e)

        show_df = st.sidebar.radio("See: ", options=["Data description", "Clean Data"])
        if show_df == "Data description":
            st.subheader("Data description")
            st.write(round(label_encode_df.describe(), 2))
            st.write(f"Rows: {label_encode_df.shape[0]}")
            st.write(f"Columns: {label_encode_df.shape[1]}")
        elif show_df == "Clean Data":
            st.subheader("Cleaned Data: ")
            st.dataframe(clean_df)
            st.write(f"Rows: {clean_df.shape[0]}")
            st.write(f"Columns: {clean_df.shape[1]}")

        model_element = st.sidebar.selectbox(
            label="Select model you want to build.",
            options=[
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVC",
                "KNN",
            ],
        )
        st.sidebar.write("Select hyperparameters:")
        if model_element == "Logistic Regression":
            penalty = st.sidebar.selectbox(label="Penalty:", options=["l2", "l1", "elasticnet"])
            solver = st.sidebar.selectbox(
                label="Solver:",
                options=[
                    "lbfgs",
                    "liblinear",
                    "newton-cg",
                    "newton-cholesky",
                    "sag",
                    "saga",
                ],
            )
            multi_class = st.sidebar.selectbox(label="Multi Class", options=["auto", "ovr", "multinomial"])

            params = {"penalty": penalty, "solver": solver, "multi_class": multi_class}

        elif model_element == "Decision Tree":
            criterion = st.sidebar.selectbox(options=["gini", "entropy", "log_loss"], label="Criterion")
            splitter = st.sidebar.selectbox(options=["best", "random"], label="splitter")
            max_depth = st.sidebar.slider(
                label="Max Depth",
                min_value=1,
                max_value=round(label_encode_df.shape[0] / 3),
            )
            min_samples_split = st.sidebar.slider(
                label="Minimum Sample Split:",
                min_value=2,
                max_value=round(label_encode_df.shape[0] / 3),
                step=5,
            )
            min_samples_leaf = st.sidebar.slider(
                label="Minimum Sample Leaf:",
                min_value=1,
                max_value=round(label_encode_df.shape[0] / 3),
                step=5,
            )
            class_weight = st.sidebar.selectbox(label="Class Weight:", options=[None, "balanced", "balanced_subsample"])
            params = {
                "criterion": criterion,
                "splitter": splitter,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": class_weight,
            }

        elif model_element == "Random Forest":
            n_estimators = st.sidebar.slider(
                label="N_Estimators:",
                min_value=1,
                max_value=round(label_encode_df.shape[0] / 3),
                step=5,
            )
            criterion = st.sidebar.selectbox(options=["gini", "entropy", "log_loss"], label="Criterion")
            splitter = st.sidebar.selectbox(options=["best", "random"], label="splitter")
            max_depth = st.sidebar.slider(
                label="Max Depth",
                min_value=1,
                max_value=round(label_encode_df.shape[0] / 3),
            )
            min_samples_split = st.sidebar.slider(
                label="Minimum Sample Split:",
                min_value=2,
                max_value=round(label_encode_df.shape[0] / 3),
                step=2,
            )
            min_samples_leaf = st.sidebar.slider(
                label="Minimum Sample Leaf:",
                min_value=1,
                max_value=round(label_encode_df.shape[0] / 3),
                step=2,
            )
            class_weight = st.sidebar.selectbox(label="Class Weight:", options=[None, "balanced", "balanced_subsample"])
            params = {
                "n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": class_weight,
            }
        elif model_element == "SVC":
            kernel = st.sidebar.selectbox(
                options=["rbf", "poly", "linear", "sigmoid"],
                label="Kernel",
            )
            gamma = st.sidebar.selectbox(options=["scale", "auto"], label="Gamma")
            class_weight = st.sidebar.selectbox(label="Class Weight:", options=[None, "balanced", "balanced_subsample"])
            decision_function_shape = st.sidebar.selectbox(options=["ovo", "ovr"], label="Decision function shape")
            params = {
                "kernel": kernel,
                "gamma": gamma,
                "decison_function_shape": decision_function_shape,
            }
        elif model_element == "KNN":
            weights = st.sidebar.selectbox(options=["uniform", "distance"], label="weights")
            algorithm = st.sidebar.selectbox(options=["auto", "ball_tree", "kd_tree", "brute"], label="algorithum")
            params = {"weights": weights, "algorithum": algorithm}

        feature_sel = st.sidebar.selectbox(
            label="Feature selection method",
            options=["None", "Correlation", "RFE", "Chi-Square"],
        )
        if feature_sel == "Correlation":
            pov_corr = st.sidebar.slider(label="Positive correlation", min_value=0.0, max_value=1.0, step=0.1)
            neg_corr = st.sidebar.slider(label="Positive correlation", min_value=0.0, max_value=-1.0, step=0.1)
            features = corr(label_encode_df, sel_target, pov_corr, neg_corr)
            if list(features.index) == []:
                st.warning(
                    "Please Select the correlation values properly or else try different feature selection method.\
                    Model will be build without feature selection."
                )
            else:
                st.subheader("Features with respect correlation:")
                if len(list(features.index)) < 5:
                    st.warning("Using less than 5 variable to build model is not recommended!")
                st.dataframe(features)
                feat_list = list(features.index)
                feat_list.append(sel_target)
                label_encode_df = label_encode_df[feat_list]
                cor = round(label_encode_df.corr(), 2)

                fig = px.imshow(cor, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

        elif feature_sel == "Chi-Square":
            features = chi(label_encode_df, sel_target)

            st.subheader("Features with respect Chi-Square:")
            if len(list(features["Feature"])) < 5:
                st.warning("Using less than 5 variable to build model is not recommended!")
            st.dataframe(features)
            feat_list = list(features["Feature"])
            feat_list.append(sel_target)
            label_encode_df = label_encode_df[feat_list]

        elif feature_sel == "RFE":
            st.subheader("Features with respect Recursive Feature Elimination :")
            num_feat = st.sidebar.slider(
                label="Select number of Feature",
                min_value=1,
                max_value=label_encode_df.shape[1] - 1,
            )
            features = rec_feat_ele(model_element, label_encode_df, sel_target, num_feat)
            if list(features[features["Imp"] == True]["Imp"]).count(True) < 5:
                st.warning("Using less than 5 variable to build model is not recommended!")
            st.dataframe(features[features["Imp"] == True])
            feat_list = list(features[features["Imp"] == True]["Feature"])
            feat_list.append(sel_target)
            label_encode_df = label_encode_df[feat_list]

        else:
            build_mod = st.sidebar.checkbox(label="Build Model")
            if build_mod:
                model_result(model_element, label_encode_df, sel_target, params)

    else:
        st.sidebar.warning("Please Upload a file.")


if __name__ == "__main__":
    main()
