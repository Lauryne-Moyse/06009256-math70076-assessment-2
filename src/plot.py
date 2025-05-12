import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


def plot_chart(df, x, y, z, hue):
    """
    Generates a chart according to the specified inputs

    Parameters:
        df (pd.DataFrame): input dataset
        x (str): name of first column
        y (str): name of second column
        z (str): name of third column
        hue (str): name of color column

    Returns:
        plotly.graph_objects.Figure: output chart
    """

    hue_col = df[hue] if hue!="---" is not None else None

    if y=="---":
        fig = single_plot(df[x], hue_col)

    elif z=="---":
        fig = pair_plot(df[x], df[y], hue_col)

    else:
        fig = triplet_plot(df[x], df[y], df[z], hue_col)

    return fig


def single_plot(x, hue):
    """
    Generates a 1-dim scatter plot according to the specified inputs

    Parameters:
        x (pd.Series): first column
        hue (pd.Series): color column

    Returns:
        plotly.graph_objects.Figure: output chart
    """

    name = hue.name if hue is not None else None

    df_plot = pd.DataFrame({
        "x": range(len(x)),
        "y":x,
        "hue": hue
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="hue", 
        title="Scatter Plot",
    )

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        legend_title_text=name,
        xaxis_title="index",
        yaxis_title=x.name,
        showlegend=hue is not None
    )

    return fig


def pair_plot(x, y, hue):
    """
    Generates a 2-dims scatter plot according to the specified inputs

    Parameters:
        x (pd.Series): first column
        y (pd.Series): second column
        hue (pd.Series): color column

    Returns:
        plotly.graph_objects.Figure: output chart
    """

    name = hue.name if hue is not None else None

    df_plot = pd.DataFrame({
        "x": x,
        "y":y,
        "hue": hue
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="hue", 
        title="Scatter Plot",
    )

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        legend_title_text=name,
        xaxis_title=x.name,
        yaxis_title=y.name,
        showlegend=hue is not None
    )

    return fig


def triplet_plot(x, y, z, hue):
    """
    Generates a 3-dims scatter plot according to the specified inputs

    Parameters:
        x (pd.Series): first column
        y (pd.Series): second column
        z (pd.Series): third column
        hue (pd.Series): color column

    Returns:
        plotly.graph_objects.Figure: output chart
    """

    name = hue.name if hue is not None else None

    df_plot = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "hue": hue
    })

    fig = px.scatter_3d(
        df_plot,
        x="x",
        y="y",
        z="z",
        color="hue", 
        title="3D Scatter Plot",
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        legend_title_text=name,
        scene=dict(
            xaxis_title=x.name,
            yaxis_title=y.name,
            zaxis_title=z.name
            ),
        showlegend=hue is not None
    )

    return fig

