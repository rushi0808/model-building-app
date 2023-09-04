import numpy as np
import pandas as pd
import ydata_profiling
from scipy.stats import chi2_contingency
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, train_test_split


def load_data(file):
    file_type = file.name.split(".")[-1]
    if file_type == "csv":
        file_data = pd.read_csv(file)
        return file_data
    elif file_type == "xlsx":
        file_data = pd.read_excel(file)
        return file_data


def profiling_rp(data):
    pr = data.profile_report()
    return pr


def datasummary(data):
    summarydf = pd.DataFrame()
    rows = data.shape[0]
    cols = data.shape[1]
    name = []
    unique = []
    null_count = []
    null_percent = []
    dtype = []
    for col in data.columns:
        name.append(col)
        unique.append(data[col].nunique())
        null_count.append(data[col].isnull().sum())
        col_null_per = round(data[col].isnull().sum() * 100 / rows, 2)
        null_percent.append(col_null_per)
        dtype.append(data[col].dtype)

    summarydf["Name"] = name
    summarydf["Unique Values"] = unique
    summarydf["Null Count"] = null_count
    summarydf["Null Percentage"] = null_percent
    summarydf["Data Type"] = dtype

    return summarydf, rows, cols


def col_drop_df(data, drop_col):
    if drop_col != []:
        col_drop_df = data.drop(columns=drop_col)
        return col_drop_df
    else:
        return data


def data_clean(df):
    drop_list = list(
        df.isnull().sum()[df.isnull().sum() * 100 / df.shape[0] > 10].index
    )
    if drop_list != []:
        df = df.drop(columns=drop_list)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].value_counts().index[0]
        elif df[col].dtype == "float64":
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype == "int64":
            df[col] = df[col].fillna(df[col].mean())

    return df


def label_encode(data):
    from sklearn.preprocessing import LabelEncoder

    data[data.select_dtypes("object").columns] = data[
        data.select_dtypes("object").columns
    ].apply(LabelEncoder().fit_transform)
    return data


def data_smapling(data, target):
    tr, ts = train_test_split(data, test_size=0.2)
    tr_x = tr.drop(target, axis=1)
    tr_y = tr[target]

    ts_x = ts.drop(target, axis=1)
    ts_y = ts[target]

    return tr_x, tr_y, ts_x, ts_y


def corr(data, lable, pov_corr, neg_corr):
    x = data.drop(columns=[lable])
    y = data[lable]
    df = pd.DataFrame(x.corrwith(y))
    df.rename(columns={0: "Importance"}, inplace=True)
    df = df[(df["Importance"] >= pov_corr) | (df["Importance"] <= neg_corr)]
    return df


def chi(data, label):
    columns = data.drop(columns=[label]).columns
    feat = []
    imp = []
    for col in columns:
        tab = pd.crosstab(data[col], data[label])
        p_val = chi2_contingency(tab)[1]
        if p_val < 0.05:
            feat.append(col)
            imp.append(round(p_val, 10))

    feat_imp = pd.DataFrame()
    feat_imp["Feature"] = feat
    feat_imp["Importance"] = imp
    return feat_imp


def rec_feat_ele(model_ele, data, target, num_feat, params=None):
    model = build_model(model_ele, params)
    x = data.drop(columns=[target])
    y = data[target]
    rfe = RFE(model, n_features_to_select=num_feat)
    rfe.fit(x, y)
    feat_imp = pd.DataFrame()
    feat_imp["Feature"] = x.columns
    feat_imp["Imp"] = rfe.support_
    return feat_imp


def coef(values, features):
    coef_df = pd.DataFrame()
    coef_df["Features"] = features
    coef_df["Weights"] = values
    return coef_df


def build_model(model_ele, params=None):
    if model_ele == "Linear Regression":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        return model

    elif model_ele == "SVR":
        from sklearn.svm import SVR

        model = SVR(kernel=params["kernel"])
        return model

    elif model_ele == "Lasso":
        from sklearn.linear_model import Lasso

        model = Lasso()
        return model

    elif model_ele == "Ridge":
        from sklearn.linear_model import Ridge

        model = Ridge()
        return model
    elif model_ele == "Decision tree":
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
        )

    elif model_ele == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty=params["penalty"],
            solver=params["solver"],
            multi_class=params["multi_class"],
        )
        return model

    elif model_ele == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(
            criterion=params["criterion"],
            splitter=params["splitter"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight=params["class_weight"],
        )
        return model

    elif model_ele == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight=params["class_weight"],
        )
        return model

    elif model_ele == "SVC":
        from sklearn.svm import SVC

        model = SVC(
            probability=True,
            kernel=params["kernel"],
            gamma=params["gamma"],
            decision_function_shape=params["decison_function_shape"],
        )
        return model

    elif model_ele == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier(
            weights=params["weights"], algorithm=params["algorithum"]
        )
        return model


def save_model(model, name):
    import pickle

    with open(f"./{name}.pkl", "wb") as file:
        pickle.dump(model, file)

    with open(f"./{name}.pkl", "rb") as file:
        model_file = file.read()

    return model_file


def del_file():
    import os

    path = r"./"
    for x in list(os.walk(path)):
        for each in x[2]:
            if each.split(".")[1] == "pkl":
                print(each)
                os.remove(each)
