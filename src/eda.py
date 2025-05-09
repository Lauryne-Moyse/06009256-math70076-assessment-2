import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def corr_heatmap(df):
    
    fig = plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    return fig


def analyse_column(col): 

    is_num = pd.api.types.is_numeric_dtype(col)

    summary = {
            "dtype": col.dtype,
            "missing": col.isna().sum(),
            "skewness": col.skew() if is_num else None,
        }

    return summary 


def plot_column(col):

    is_num = pd.api.types.is_numeric_dtype(col)

    fig = plt.figure()

    if is_num:
        sns.histplot(col.dropna(), kde=True)
    else:
        col.value_counts().plot(kind="bar")
    
    return fig
    

