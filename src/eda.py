import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def corr_heatmap(df):
    """
    Produces Pearson correlation heatmap of df features 

    Parameters:
        df (pd.DataFrame): input dataset

    Returns:
        plt.Figure: features correlation heatmap
    """
    
    dim = min(0.5*len(df.columns), 20)
    fig = plt.figure(figsize=(dim, dim))
    
    sns.heatmap(
        df.corr(), 
        annot=True, 
        cmap='coolwarm',
        fmt=".3f",
        annot_kws={"size": dim+2},
    )

    return fig


def analyse_column(col, type): 
    """
    Produces a short summary of the specified column

    Parameters:
        col (pd.Series): input column
        type (str): type of the column

    Returns:
        dictionary: summary of the column
    """

    summary = {
        "dtype": col.dtype,
        "missing": col.isna().sum(),
        "skewness": col.skew() if type=="continuous" else None
        }

    return summary 


def plot_column(col, type):
    """
    Produces a plot of the distribution of the specified column:
    - bar plot for categorical type 
    - histogram for continuous type

    Parameters:
        col (pd.Series): input column
        type (str): type of the column

    Returns:
        plt.Figure: distribution plot
    """

    fig = plt.figure()

    if type=="continuous":

        # Histogram
        fig, ax = plt.subplots()

        sns.histplot(
            col.dropna(), 
            bins="fd", 
            kde=True, 
            ax=ax, 
            edgecolor='black'
        )

        ax.set_title(f"Histogram of {col.name}")
        ax.set_xlabel(col.name)
        ax.set_ylabel("Count")

    else:
        
        # Bar plot
        fig, ax = plt.subplots()
    
        col.value_counts().plot(
            kind="bar", 
            ax=ax, 
            color='lightcoral', 
            edgecolor='black'
        )
        
        ax.set_title(f"Bar plot of {col.name}")
        ax.set_xlabel(col.name)
        ax.set_ylabel("Count")
    
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        
    return fig
    

