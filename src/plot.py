import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


def plot_chart(df, x, y, z, hue):

    hue_col = df[hue] if hue!="---" is not None else None

    if y=="---":
        fig = single_plot(df[x], hue_col)

    elif z=="---":
        fig = pair_plot(df[x], df[y], hue_col)

    else:
        fig = triplet_plot(df[x], df[y], df[z], hue_col)

    return fig


# Plot a 1-dimensional graph
def single_plot(x, hue):

    name = hue.name if hue is not None else None

    df_plot = pd.DataFrame({
        "x": range(len(x)),
        "y":x,
        "hue": hue if hue is not None else "All"
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


# Plot a 2-dimensional graph
def pair_plot(x, y, hue):

    name = hue.name if hue is not None else None

    df_plot = pd.DataFrame({
        "x": x,
        "y":y,
        "hue": hue if hue is not None else "All"
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


# Plot a 3-dimensional graph
def triplet_plot(x, y, z, hue):

    name = hue.name if hue is not None else None

    df_plot = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "hue": hue if hue is not None else "All"
    })

    fig = px.scatter_3d(
        df_plot,
        x="x",
        y="y",
        z="z",
        color="hue", 
        title="3D Scatter Plot",
    )

    fig.update_traces(marker=dict(size=3))
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
