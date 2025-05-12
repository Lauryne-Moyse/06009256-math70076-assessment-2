import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def corr_heatmap(df):
    
    fig = plt.figure()
    
    sns.heatmap(
        df.corr(), 
        annot=True, 
        cmap='coolwarm'
    )

    return fig


def analyse_column(col, type): 

    summary = {
            "dtype": col.dtype,
            "missing": col.isna().sum(),
            "skewness": col.skew() if type=="continuous" else None
        }

    return summary 


def plot_column(col, type):


    fig = plt.figure()

    if type=="continuous":

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
    

