import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from plotly.graph_objects import Figure
from statsmodels.iolib.summary import Summary

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import ML


### TEST one_hot_warning ###

def test_one_hot_warning():
    df_test = pd.DataFrame({
        'Category1': ['A', 'B', 'B', 'A', 'A', 'B'],
        'Category2': ['X', 'Y', 'Z', 'X', 'Y', 'Z']
    })
    to_one_hot_cols = ['Category1', 'Category2']
    warning_cols_names, warning_cols_length = ML.one_hot_warning(df_test, to_one_hot_cols, 2)
    assert 'Category2' in warning_cols_names
    assert len(warning_cols_names) == 1  
    assert warning_cols_length[0] == 3  


### TEST get_df_encoded ###

def test_get_df_encoded(): 
    df_test = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'Category1': ['A', 'B', 'C', 'A', 'E'],
        'Category2': ['AB', 'BC', 'CA', 'AD', 'ED'],
        'Category3': ['blue', 'orange', 'green', 'yellow', 'pink']
    })
    to_label_cols = ['Category1', 'Category2']
    to_one_hot_cols = ['Category3']
    df_encoded, mappings = ML.get_df_encoded(df_test, to_label_cols, to_one_hot_cols)
    assert isinstance(df_encoded['Category1'].iloc[0], np.number)
    assert isinstance(df_encoded['Category2'].iloc[0], np.number)
    assert 'Category3_orange' in df_encoded.columns
    assert 'feature1' in df_encoded.columns
    assert isinstance(mappings, pd.DataFrame)

def test_get_df_encoded_no_label(): 
    df_test = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'Category1': ['A', 'B', 'C', 'A', 'E'],
        'Category2': ['AB', 'BC', 'CA', 'AD', 'ED'],
        'Category3': ['blue', 'orange', 'green', 'yellow', 'pink']
    })
    to_label_cols = []
    to_one_hot_cols = ['Category2', 'Category3']
    df_encoded, mappings = ML.get_df_encoded(df_test, to_label_cols, to_one_hot_cols)
    assert 'Category2_BC' in df_encoded.columns
    assert 'Category3_orange' in df_encoded.columns
    assert 'feature1' in df_encoded.columns
    assert 'Category1' in df_encoded.columns
    assert mappings == {}


### TEST plot_pred ###

def test_plot_pred():
    y_test =  [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    y_pred =  [5, 4, 3, 2, 5, 1, 2, 3, 4, 5]
    fig, rmse = ML.plot_pred("target", y_test, y_pred)
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert rmse >= 0


### TESTS run_model ###

def test_run_model():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [2, 3, 4, 5, 6]
    })
    target = 'target'
    features = ['feature1', 'feature2']
    model = 'OLS'
    type = 'regression'
    split = 0.8
    result = ML.run_model(df, target, features, model, type, split)
    assert isinstance(result, tuple)
    assert len(result) == 4
    summary, fig, rmse, mae = result
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse >= 0
    assert mae >= 0

def test_run_model():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    })
    target = 'target'
    features = ['feature1', 'feature2']
    model = 'Logistic'
    type = 'classification'
    split = 0.8
    result = ML.run_model(df, target, features, model, type, split)
    assert isinstance(result, tuple)
    assert len(result) == 6
    summary, fig, acc, prec, rec, f1 = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(acc, float)
    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert isinstance(f1, float)
    assert acc >= 0
    assert prec >= 0
    assert rec >= 0
    assert f1 >= 0


### TEST run_ols ###

def test_run_ols():
    X, y = make_regression()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, rmse, mae = ML.run_ols(
        target, X_train, X_test, y_train, y_test
    )
    assert isinstance(summary, Summary)
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse >= 0
    assert mae >= 0


### TEST run_lasso ###

def test_run_lasso():
    X, y = make_regression()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, rmse, mae = ML.run_lasso(
        target, features, X_train, X_test, y_train, y_test
    )
    assert isinstance(summary, str)
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse >= 0
    assert mae >= 0


### TEST run_ridge ###

def test_run_ridge():
    X, y = make_regression()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, rmse, mae = ML.run_ridge(
        target, X_train, X_test, y_train, y_test
    )
    assert isinstance(summary, str)
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse >= 0
    assert mae >= 0


### TESTS run_rf ###

def test_run_rfr():
    X, y = make_regression()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, rmse, mae = ML.run_rfr(
        target, X_train, X_test, y_train, y_test
    )
    assert isinstance(summary, str)
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse >= 0
    assert mae >= 0

def test_run_rfc():
    X, y = make_classification(n_classes=3, n_informative=3)
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, acc, prec, rec, f1 = ML.run_rfc(
        X_train, X_test, y_train, y_test
    )
    assert isinstance(summary, str)
    assert isinstance(fig, plt.Figure)
    assert isinstance(acc, float)
    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert isinstance(f1, float)
    assert acc >= 0
    assert prec >= 0
    assert rec >= 0
    assert f1 >= 0


### TEST run_xgb ###

def test_run_xgbr():
    X, y = make_regression()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    metrics_plots, fig, rmse, mae = ML.run_xgbr(
        target, X_train, X_test, y_train, y_test
    )
    assert isinstance(metrics_plots, plt.Figure)
    assert isinstance(fig, Figure)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse >= 0
    assert mae >= 0

def test_run_xgbc():
    X, y = make_classification(n_classes=3, n_informative=3)
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, acc, prec, rec, f1 = ML.run_xgbc(
        X_train, X_test, y_train, y_test
    )
    assert len(fig) == 2
    matrix, metrics_plots = fig
    assert isinstance(summary, str)
    assert isinstance(matrix, plt.Figure)
    assert isinstance(metrics_plots, plt.Figure)
    assert isinstance(acc, float)
    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert isinstance(f1, float)
    assert acc >= 0
    assert prec >= 0
    assert rec >= 0
    assert f1 >= 0


### TEST plot_metrics_xgb ###

def test_plot_metrics_xgb():
    X, y = make_regression()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    booster = model.get_booster()
    fig = ML.plot_metrics_xgb(booster)
    assert isinstance(fig, plt.Figure)


### TEST run_logistic ###
def test_run_logisitc():
    X, y = make_classification(n_classes=3, n_informative=3)
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    target = "target"
    df[target] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=1
    )
    summary, fig, acc, prec, rec, f1 = ML.run_logistic(
        X_train, X_test, y_train, y_test
    )
    assert isinstance(summary, str)
    assert isinstance(fig, plt.Figure)
    assert isinstance(acc, float)
    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert isinstance(f1, float)
    assert acc >= 0
    assert prec >= 0
    assert rec >= 0
    assert f1 >= 0
