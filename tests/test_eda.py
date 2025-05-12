import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import eda


### TEST corr_heatmap ###

def test_corr_heatmap():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": [4, 3, 2, 1],
        "d": ["red", "orange", "green", "blue"]
    })
    fig = eda.corr_heatmap(df)
    assert isinstance(fig, plt.Figure)


### TESTS analyse_column ###

def test_analyse_column_numeric():
    s = pd.Series([1, None, 3, None, 5], name="num_col")
    result = eda.analyse_column(s)
    assert result["dtype"] == s.dtype
    assert result["missing"] == 2
    assert isinstance(result["skewness"], float)

def test_analyse_column_categorical():
    s = pd.Series(["a", "b", "a", "c"], name="cat_col")
    result = eda.analyse_column(s)
    assert result["skewness"] is None


### TESTS plot_column ###

def test_plot_column_numeric():
    s = pd.Series([1, 2, 3, 4], name="plot_num")
    fig = eda.plot_column(s)
    assert isinstance(fig, plt.Figure)

def test_plot_column_categorical():
    s = pd.Series(["A", "B", "A", "C"], name="plot_cat")
    fig = eda.plot_column(s)
    assert isinstance(fig, plt.Figure)
