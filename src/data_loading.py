import pandas as pd


def load_case1():
    train_data = pd.read_csv("data/case1/train1.csv")
    test_data = pd.read_csv("data/case1/test1.csv")
    for data in (train_data, test_data):
        data["speed_krpm"] = data["speed_rpm"] / 1000
        data["pressure_rise_bar"] = data["outlet_pressure_bar"] - data["inlet_pressure_bar"]
    return train_data, test_data