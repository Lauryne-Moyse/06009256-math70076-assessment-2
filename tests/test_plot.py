import numpy as np
import pandas as pd
from plotly.graph_objects import Figure

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import plot


### TESTS single_plot ###

def test_singe_plot_no_hue():
    x = pd.Series([i for i in range(50)], name="x")
    hue = None
    fig = plot.single_plot(x, hue)
    assert isinstance(fig, Figure)

def test_singe_plot_cat_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "hue": ["red", "orange", "green", "blue", "brown"]*10
    })
    fig = plot.single_plot(df["x"], df["hue"])
    assert isinstance(fig, Figure)

def test_singe_plot_numcat_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "hue": [1.0, 2.0, 3.0, 4.0, 5.0]*10
    })
    fig = plot.single_plot(df["x"], df["hue"])
    assert isinstance(fig, Figure)

def test_singe_plot_continuous_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "hue": np.linspace(1, 5, 50)
    })
    fig = plot.single_plot(df["x"], df["hue"])
    assert isinstance(fig, Figure)


### TESTS pair_plot ###

def test_pair_plot_no_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)]
    })
    hue = None
    fig = plot.pair_plot(df["x"], df["y"], hue)
    assert isinstance(fig, Figure)

def test_pair_plot_cat_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "hue": ["red", "orange", "green", "blue", "brown"]*10
    })
    fig = plot.pair_plot(df["x"], df["y"], df["hue"])
    assert isinstance(fig, Figure)

def test_pair_plot_numcat_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "hue": [1.0, 2.0, 3.0, 4.0, 5.0]*10
    })
    fig = plot.pair_plot(df["x"], df["y"], df["hue"])
    assert isinstance(fig, Figure)

def test_pair_plot_continuous_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "hue": np.linspace(1, 5, 50)
    })
    fig = plot.pair_plot(df["x"], df["y"], df["hue"])
    assert isinstance(fig, Figure)

### TESTS triplet_plot ###

def test_triplet_plot_no_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "z": np.linspace(0, 100, 50)
    })
    hue = None
    fig = plot.triplet_plot(df["x"], df["y"], df["z"], hue)
    assert isinstance(fig, Figure)

def test_triplet_plot_cat_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "z": np.linspace(0, 100, 50),
        "hue": ["red", "orange", "green", "blue", "brown"]*10
    })
    fig = plot.triplet_plot(df["x"], df["y"], df["z"], df["hue"])
    assert isinstance(fig, Figure)

def test_triplet_plot_numcat_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "z": np.linspace(0, 100, 50),
        "hue": [1.0, 2.0, 3.0, 4.0, 5.0]*10
    })
    fig = plot.triplet_plot(df["x"], df["y"], df["z"], df["hue"])
    assert isinstance(fig, Figure)

def test_triplet_plot_continuous_hue():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "z": np.linspace(0, 100, 50),
        "hue": np.linspace(1, 5, 50)
    })
    fig = plot.triplet_plot(df["x"], df["y"], df["z"], df["hue"])
    assert isinstance(fig, Figure)


### TEST plot_chart ###

def test_plot_chart():
    df = pd.DataFrame({
        "x": [i for i in range(50)],
        "y": [i for i in range(50, 100)],
        "z": np.linspace(0, 100, 50),
        "hue": np.linspace(1, 5, 50)
    })
    fig = plot.plot_chart(df, "x", "y", "---", "hue")
    assert isinstance(fig, Figure)