import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import main


def test_clean_dataset_for_app():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": [4, 3, 2, 1],
        "d": ["red", "orange", "green", "blue"],
        "e": ["A", "B", "C", "B"]
    })
    numeric_cols, categorical_cols, problematic_cols = main.clean_dataset_for_app(df, 3)
    assert len(numeric_cols) == 3
    assert len(categorical_cols) == 1
    assert len(problematic_cols) == 1