import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import statsmodels.api as sm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



#===== GENERAL FUNCTIONS =====# 

def one_hot_warning(df, cols, threshold):
    """
    Identifies column with high dimensionality for one-hot encoding

    Parameters:
        df (pd.DataFrame): input dataset
        cols (list): columns to one-hot encode
        threshold (int): limit value for low dimensionality

    Returns:
        list: high-dimensionality columns 
        list: dimensions of identified at risk columns 
    """

    warning_cols_names = []
    warning_cols_length = []

    for col in cols:

        n = df[col].nunique()
        if n > threshold:
            warning_cols_names.append(col)
            warning_cols_length.append(n)
    
    return warning_cols_names, warning_cols_length


def get_df_encoded(df, to_label_cols, to_one_hot_cols):   
    """
    Identifies column with high dimensionality for one-hot encoding

    Parameters:
        df (pd.DataFrame): input dataset
        to_label_cols (list): columns to label encode
        to_one_hot_cols (int): columns to one-hot encode

    Returns:
        pd.DataFrame: updated df with encoded variables
        list: original-new values mapping for label encoding
    """

    mappings = {}

    # Label encoding
    for col in to_label_cols: 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # Mapping of the encoding 
        col1 = col+"_original"
        col2 = col+"_encoded"
        mappings[col1] = list(le.classes_)
        mappings[col2] =  list(le.transform(le.classes_))

    if len(to_label_cols) > 0:
        max_len = max(len(v) for v in mappings.values())
        # Make all the columns the same length
        for k, v in mappings.items():
            if len(v) < max_len:
                padding = [None] * (max_len - len(v))  
                mappings[k] += padding
        mappings = pd.DataFrame(mappings)

    # One-hot encoding
    for col in to_one_hot_cols: 
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    return df, mappings


# Plot y_pred against y_test
def plot_pred(target, y_test, y_pred):
    """
   Generates scatter plot of predicted against original values

    Parameters:
        target (str): name of target variable 
        y_test (np.array): original test values
        y_pred (np.array): predicted test values

    Returns:
        ploty.graph_objects.Figure: generated plot
        float: root mean squared error of the prediction
    """

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    df_plot = pd.DataFrame({
        "x": y_test,
        "y": y_pred,
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='steelblue')
    ))

    min_val = min(df_plot.min())
    max_val = max(df_plot.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(dash='dash', color='red'),
        showlegend=False
    ))

    fig.update_layout(
        title=f"Prediction of {target} (RMSE = {rmse:.2f})",
        xaxis_title="Real values",
        yaxis_title="Predicted values",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False
    )

    return fig, rmse


# Split data and run model according to the value of split
def run_model(df, target, features, model, type, split):
    """
    Split data and run model as specified

    Parameters:
        df (pd.Dataframe): input dataset
        target (str): name of regressive variable
        features (list): names of explanatory variables
        model_type (str): machine learning model
        split (float): percentage of train_size

    Returns:
        str or statsmodels.iolib.summary.Summary or plt.Figure: summary of the model
        ploty.graph_objects.Figure:  scatter plot of predicted against original values
        float: rmse of the prediction
        float: mae of the prediction 
    """

    # Split data 
    X = df[features].dropna()
    y = df[target].dropna()
    X, y = X.align(y, join='inner', axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)

    if type == "regression":
        if model == "OLS":
            res, fig, rmse, mae = run_ols(target, X_train, X_test, y_train, y_test)

        elif model == "Lasso":       
            res, fig, rmse, mae = run_lasso(target, features, X_train, X_test, y_train, y_test)

        elif model == "Ridge":       
            res, fig, rmse, mae = run_ridge(target, X_train, X_test, y_train, y_test)

        elif model == "Random-Forest-r":       
            res, fig, rmse, mae = run_rfr(target, X_train, X_test, y_train, y_test)

        elif model == "XGBoost-r":       
            res, fig, rmse, mae = run_xgbr(target, X_train, X_test, y_train, y_test)

        return res, fig, rmse, mae
    
    else: 
        if model == "Logistic":
            res, fig, acc, prec, rec, f1 = run_logistic(X_train, X_test, y_train, y_test)

        elif model == "Random-Forest-c":        
            res, fig, acc, prec, rec, f1 = run_rfc(X_train, X_test, y_train, y_test)

        elif model == "XGBoost-c":        
            res, fig, acc, prec, rec, f1 = run_xgbc(X_train, X_test, y_train, y_test)

        return res, fig, acc, prec, rec, f1



#===== REGRESSION MODELS TRAINING =====#

def run_ols(target, X_train, X_test, y_train, y_test):
    """
    Fit an OLS regression

    Parameters:
        target (str): name of regressive variable 
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        statsmodels.iolib.summary.Summary: summary of the model
        ploty.graph_objects.Figure:  scatter plot of predicted against original values
        float: rmse of the prediction
        float: mae of the prediction 
    """

    # Model training
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test, has_constant='add')

    model = sm.OLS(y_train, X_train_const).fit()
    y_pred = model.predict(X_test_const)
    mae = mean_absolute_error(y_test, y_pred)

    # Summary
    summary = model.summary()

    # Scatter predictions
    fig, rmse = plot_pred(target, y_test, y_pred)

    return summary, fig, rmse, mae


def run_lasso(target, features, X_train, X_test, y_train, y_test):
    """
    Fit a Lasso regression

    Parameters:
        target (str): name of regressive variable 
        features (list): names of explanatory variables 
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        str: summary of the model
        ploty.graph_objects.Figure:  scatter plot of predicted against original values
        float: rmse of the prediction
        float: mae of the prediction 
    """

    # Model training
    model = Lasso()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Scatter plot of predictions
    fig, rmse = plot_pred(target, y_test, y_pred)

    # Summary 
    non_zero = np.sum(model.coef_ != 0)
    selected_vars = [f for f, c in zip(features, model.coef_) if c != 0]
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)

    summary = f"""  
    - Number of selected features: {non_zero} out of {len(features)}  
    - Selected features: {', '.join(selected_vars)}  
    - MAE on test set: {mae:.4f} 
    - RMSE on test set: {rmse:.4f} 
    - R² on test set: {r2:.4f}  
    """

    return summary, fig, rmse, mae


def run_ridge(target, X_train, X_test, y_train, y_test):
    """
    Fit a Ridge regression

    Parameters:
        target (str): name of regressive variable 
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        str: summary of the model
        ploty.graph_objects.Figure:  scatter plot of predicted against original values
        float: rmse of the prediction
        float: mae of the prediction 
    """

    # Model training
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Scatter plot of predictions
    fig, rmse = plot_pred(target, y_test, y_pred)

    # Summary
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)

    summary = f"""
    - MAE on test set: {mae:.4f} 
    - RMSE on test set: {rmse:.4f} 
    - R² on test set: {r2:.4f}  
    """

    return summary, fig, rmse, mae


def run_rfr(target, X_train, X_test, y_train, y_test):
    """
    Fit a random forest regression

    Parameters:
        target (str): name of regressive variable 
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        str: summary of the model
        ploty.graph_objects.Figure:  scatter plot of predicted against original values
        float: rmse of the prediction
        float: mae of the prediction 
    """

    # Model training
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Scatter plot of predictions
    fig, rmse = plot_pred(target, y_test, y_pred)

    # Summary
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = model.score(X_test, y_test)

    summary = f"""
    - MAE on test set: {mae:.4f} 
    - RMSE on test set: {rmse:.4f} 
    - R² on test set: {r2:.4f}  
    """

    return summary, fig, rmse, mae


def run_xgbr(target, X_train, X_test, y_train, y_test):
    """
    Fit an XGBoost regression 

    Parameters:
        target (str): name of regressive variable 
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        plt.Figure: plot of the model's metrics 
        ploty.graph_objects.Figure:  scatter plot of predicted against original values
        float: rmse of the prediction
        float: mae of the prediction 
    """

    # Model training
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Scatter plot of predictions
    fig, rmse = plot_pred(target, y_test, y_pred)

    # Metrics plots 
    metrics_plots = plot_metrics_xgb(model.get_booster())

    return metrics_plots, fig, rmse, mae


def plot_metrics_xgb(booster):
    """
    Generate a plot of XGBoost model metrics 

    Parameters:
        booster (xgb.core.Booster): parameters of XGBoost model 

    Returns:
        plt.Figure: plots of main trained features of XGBoost model 
    """

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    for i, metric in enumerate(["gain", "weight", "cover"]):

        features_importance = booster.get_score(importance_type=metric)
        df_importance = pd.DataFrame(
            sorted(features_importance.items(), key=lambda x: x[1]),
            columns=['Feature', 'Importance']
        )

        color = tuple(1 if j==i else 0 for j in range(3))
        ax[i].barh(df_importance['Feature'], df_importance['Importance'], color=color)
        ax[i].set_xlabel(metric)
        ax[i].set_ylabel('Features')

    ax[0].set_title('Impact on predictive power by feature')
    ax[1].set_title('Frequency of use by feature')
    ax[2].set_title('Cover of divisions by feature')

    plt.tight_layout()

    return fig



#===== Classification MODELS TRAINING =====#

def run_logistic(X_train, X_test, y_train, y_test):
    """
    Fit a logistic regression classifier

    Parameters:
        features (list): name of explanatory variables
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        str: text summary of performance metrics
        plt.Figure: confusion matrix plot
        float: accuracy score
        float: precision score
        float: recall score
        float: F1 score
    """
    
    # Model training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Summary
    summary = classification_report(y_test, y_pred, output_dict=False)

    # Metrics
    nb_class = len(set(np.concatenate((y_test, y_train))))
    if nb_class > 2:
        average_method = "weighted"
    else:
        average_method = "binary"
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average_method)
    rec = recall_score(y_test, y_pred, average=average_method)
    f1 = f1_score(y_test, y_pred, average=average_method)

    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    class_labels = sorted(set(np.concatenate((y_train, y_test)))) 
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    return summary, fig, acc, prec, rec, f1



def run_rfc(X_train, X_test, y_train, y_test):
    """
    Fit a random forest classifier

    Parameters:
        features (list): name of explanatory variables
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        str: text summary of performance metrics
        plt.Figure: confusion matrix plot
        float: accuracy score
        float: precision score
        float: recall score
        float: F1 score
    """
    
    # Model training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Summary
    summary = classification_report(y_test, y_pred, output_dict=False)

    # Metrics
    nb_class = len(set(np.concatenate((y_test, y_train))))
    if nb_class > 2:
        average_method = "weighted"
    else:
        average_method = "binary"
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average_method)
    rec = recall_score(y_test, y_pred, average=average_method)
    f1 = f1_score(y_test, y_pred, average=average_method)

    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    class_labels = sorted(set(np.concatenate((y_train, y_test)))) 
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    return summary, fig, acc, prec, rec, f1


def run_xgbc(X_train, X_test, y_train, y_test):
    """
    Fit a XGBoost classifier

    Parameters:
        X_train (np.array): training set for explanatory variables
        X_test (np.array): test set for explanatory variables
        y_train (np.array): training set for regressive variable
        y_test (np.array): test set for regressive variable

    Returns:
        str: text summary of performance metrics
        tuple: confusion matrix and booster metrics plots
        float: accuracy score
        float: precision score
        float: recall score
        float: F1 score
    """
    
    # Model training
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Summary
    summary = classification_report(y_test, y_pred, output_dict=False)

    # Metrics
    nb_class = len(set(np.concatenate((y_test, y_train))))
    if nb_class > 2:
        average_method = "weighted"
    else:
        average_method = "binary"
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average_method)
    rec = recall_score(y_test, y_pred, average=average_method)
    f1 = f1_score(y_test, y_pred, average=average_method)

    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    class_labels = sorted(set(np.concatenate((y_train, y_test)))) 
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    booster = model.get_booster()
    metrics_plots = plot_metrics_xgb(booster)

    return summary, (fig, metrics_plots), acc, prec, rec, f1