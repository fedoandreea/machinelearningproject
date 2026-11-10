"""Shared data loading and feature engineering.

All notebooks should load data through these functions, so every task uses
the same files and the same engineered features.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = "flow_m3_h"


def load_case1():
    """Return (train, test) DataFrames for Case 1 (480 / 120 rows)."""
    train = pd.read_csv(DATA_DIR / "case1" / "train1.csv")
    test = pd.read_csv(DATA_DIR / "case1" / "test1.csv")
    return train, test


def load_case2():
    """Return (train, test) DataFrames for Case 2 (59,950 / 119,900 rows).

    Train and test contain different pumps -> use GroupKFold with groups=train["pump_id"].
    """
    train = pd.read_csv(DATA_DIR / "case2" / "train2.csv")
    test = pd.read_csv(DATA_DIR / "case2" / "test2.csv")
    return train, test


def add_features(df):
    """Add the unit-converted / combined features used in Task 1 (returns a copy)."""
    df = df.copy()
    df["speed_krpm"] = df["speed_rpm"] / 1000
    df["power_kw"] = df["power_w"] / 1000
    df["pressure_difference_bar"] = df["outlet_pressure_bar"] - df["inlet_pressure_bar"]
    return df
