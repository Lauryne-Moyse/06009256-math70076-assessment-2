import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error



#===== GENERAL FUNCTIONS =====# 


def get_df_encoded(df, to_label_cols, to_one_hot_cols): 

    for col in to_label_cols: 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    for col in to_one_hot_cols: 
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df


# Split data and run model according to the value of split
def run_model(df, target, features, model_type, split):

    # Splitting data 
    X = df[features].dropna()
    y = df[target].dropna()
    X, y = X.align(y, join='inner', axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)

    if model_type == "OLS":
        
        res, fig, rmse, mae = run_OLS(target, X_train, X_test, y_train, y_test)

    elif model_type == "Lasso":
        
        res, fig, rmse, mae = run_lasso(target, features, X_train, X_test, y_train, y_test)

    elif model_type == "Ridge":
        
        res, fig, rmse, mae = run_ridge(target, X_train, X_test, y_train, y_test)

    elif model_type == "Random Forest":
        
        res, fig, rmse, mae = run_rf(target, X_train, X_test, y_train, y_test)

    elif model_type == "XGBoost":
         
        res, fig, rmse, mae = run_xgb(target, features, X_train, X_test, y_train, y_test)

    return res, fig, rmse, mae


# Plot y_pred against y_test
def plot_pred(target, y_test, y_pred):

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    df_plot = pd.DataFrame({
        "x": y_test,
        "y": y_pred,
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["x"],
        y=df_plot["y"],
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



#===== MODELS TRAINING =====#

def run_OLS(target, X_train, X_test, y_train, y_test):

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


def run_rf(target, X_train, X_test, y_train, y_test):

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


def run_xgb(target, features, X_train, X_test, y_train, y_test):

    # Model training
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Scatter plot of predictions
    fig, rmse = plot_pred(target, y_test, y_pred)

    # Metrics plots 
    metrics_plots = plot_metrics_xgb(features, model.get_booster())


    return metrics_plots, fig, rmse, mae


def plot_metrics_xgb(features, booster):

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




