import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

from src.functions import (
    build_model,
    chi,
    coef,
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


def model_result(model_element, x_labels, target, param=None):
    try:
        st.title(f"Model {model_element} Results:")
        tr_x, tr_y, ts_x, ts_y = data_smapling(x_labels, target)
        model_built = build_model(model_element, param)
        model_built.fit(tr_x, tr_y)
        rsq_tr = model_built.score(tr_x, tr_y)
        rsq_ts = model_built.score(ts_x, ts_y)
        pred_tr = model_built.predict(tr_x)
        pred_ts = model_built.predict(ts_x)
        err_tr = tr_y - pred_tr
        err_ts = ts_y - pred_ts

        st.subheader("Train scores: ")
        coef_df = coef(model_built.coef_, model_built.feature_names_in_)
        st.data_editor(coef_df)
        st.markdown(f"**Rsquare: {round(rsq_tr,2)}**")
        st.markdown(
            f"**MSE: {round(np.mean(np.square(err_tr)),2)} | RMSE: {round(np.sqrt(np.mean(np.square(err_tr))),2)}**"
        )
        st.markdown(
            f"**MAPE: {round(np.mean(np.abs(err_tr*100/tr_y)),2)} | Accuracy: {100 - round(np.mean(np.abs(err_tr*100/tr_y)),2)}**"
        )
        st.write(15 * "-")
        st.subheader("Test scores: ")
        st.markdown(f"**Rsquare: {round(rsq_ts,2)}**")
        st.markdown(
            f"**MSE: {round(np.mean(np.square(err_ts)),2)} | RMSE: {round(np.sqrt(np.mean(np.square(err_ts))),2)}**"
        )
        st.markdown(
            f"**MAPE: {round(np.mean(np.abs(err_ts*100/ts_y)),2)} | Accuracy: {100 - round(np.mean(np.abs(err_ts*100/ts_y)),2)}**"
        )
        st.write(15 * "-")
        st.subheader("Assumptions done on train data: ")
        st.markdown(
            f"**Mean: {round(err_tr.mean(), 2)}         |   Median: {round(err_tr.median(), 2)}**"
        )
        st.markdown(
            f"**Kurtosis: {round(err_tr.kurtosis(), 2)} |   Skewness: {round(err_tr.skew(), 2)}**"
        )
        st.write(15 * "-")
        st.subheader("Scatter Plot of errors:")
        fig = px.scatter(err_tr)
        st.plotly_chart(fig)
        fig = px.histogram(err_tr)
        st.write(15 * "-")
        st.subheader("Histogram Plot of errors:")
        st.plotly_chart(fig)
        fig = px.box(err_tr)
        st.write(15 * "-")
        st.subheader("Box Plot of errors:")
        st.plotly_chart(fig)
        fig = px.scatter(x=tr_y, y=pred_tr, trendline="ols")
        st.write(15 * "-")
        st.subheader("Regression Plot of errors:")
        st.plotly_chart(fig)
        st.sidebar.success("Model build Sucessfully!")
        # making model_zip and downloading model
        save_model(model_built)
        model_zip()
        st.sidebar.download_button(
            label="Download model",
            data=open(r"./model_zip.zip", "rb").read(),
            file_name="model_zip.zip",
        )
        st.success("Model downloaded!")
        del_file()

    except Exception as e:
        st.warning(e)


def regression():
    st.title("Regression App")
    st.sidebar.title("Options: ")
    file = st.sidebar.file_uploader(
        "Choose a file", type=["csv", "xlsx"], help="Upload CSV or Excel file."
    )

    if file:
        data = load_data(file)
        data_option = st.sidebar.radio(
            label="Go to: ", options=["Dataframe", "Profile Report"]
        )
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
            label="Select columns to drop.",
            options=data.columns,
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
                help="Select variable that you want to predict make sure that the variable is continous.",
            )

            df = data
            clean_df = data_clean(df)
            label_encode_df = label_encode(clean_df)

        col_to_trans = st.sidebar.multiselect(
            label="Select columns to transform.",
            options=[col for col in num_col if col not in col_drop],
            help="Select the columns on which you want to apply StandarScaling, min max scaling or log tranformation.",
        )
        trans_method = st.sidebar.radio(
            label="Select transform method:",
            options=["None", "StandarScale", "Min Max Scaler", "Log transformation"],
            help="Select the method of data transformation as best fit for your data.",
        )

        if trans_method != "None":
            label_encode_df = data_transform(
                trans_method, label_encode_df, col_to_trans
            )

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
            options=["Linear Regression", "Lasso", "Ridge", "SVR"],
        )

        if model_element == "SVR":
            kernel = st.sidebar.selectbox(
                label="Kernel:",
                options=["rbf", "linear", "poly", "sigmoid"],
            )
            param = {"kernel": kernel}
        feature_sel = st.sidebar.selectbox(
            label="Feature selection method",
            options=["None", "Correlation", "RFE", "Chi-Square"],
        )

        if feature_sel == "Correlation":
            pov_corr = st.sidebar.slider(
                label="Positive correlation", min_value=0.0, max_value=1.0, step=0.1
            )
            neg_corr = st.sidebar.slider(
                label="Positive correlation", min_value=0.0, max_value=-1.0, step=0.1
            )
            features = corr(label_encode_df, sel_target, pov_corr, neg_corr)
            if not list(features.index):
                st.warning(
                    "Please Select the correlation values properly or else try different feature selection method.\
                    Model will be build without feature selection."
                )
            else:
                st.subheader("Features with respect correlation:")
                if len(list(features.index)) < 5:
                    st.warning(
                        "Using less than 5 variable to build model is not recommended!"
                    )
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
                st.warning(
                    "Using less than 5 variable to build model is not recommended!"
                )
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
            features = rec_feat_ele(
                model_element, label_encode_df, sel_target, num_feat
            )
            if list(features[features["Imp"] == True]["Imp"]).count(True) < 5:
                st.warning(
                    "Using less than 5 variable to build model is not recommended!"
                )
            st.dataframe(features[features["Imp"] == True])
            feat_list = list(features[features["Imp"] == True]["Feature"])
            feat_list.append(sel_target)
            label_encode_df = label_encode_df[feat_list]

        else:
            build_mod = st.sidebar.checkbox(label="Build Model")
            if build_mod:
                if model_element == "SVR":
                    model_result(
                        model_element,
                        label_encode_df,
                        sel_target,
                        param=param,
                    )
                else:
                    model_result(model_element, label_encode_df, sel_target)

    else:
        st.sidebar.warning("Please Upload a file.")
