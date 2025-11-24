import os
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm


def treat_timestamp(raw_date: int) -> datetime:
    str_date = str(raw_date)
    year = str_date[0:4]
    month = str_date[4:6]
    day = str_date[6:8]
    hour = str_date[8:10]
    minute = str_date[10:12]

    date_str = f"{year}-{month}-{day} {hour}:{minute}"

    return datetime.strptime(date_str, "%Y-%m-%d %H:%M")


def treat_candle_data(path="data/b3_candles_raw.csv") -> pd.DataFrame:
    raw_data = pd.read_csv(path, sep=";")
    data = pd.DataFrame()

    data = raw_data.loc[:, ~raw_data.columns.str.startswith("UNK")].copy()

    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], format="%Y%m%d%H%M").copy()

    return data


def load_ipea_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    return df.set_index("DATE").sort_index()


def load_normalize_ipea_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    df["DATE"] = df["DATE"].dt.to_period("M").dt.to_timestamp("M")
    return df.set_index("DATE").sort_index()


def load_normalize_ibc_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["DATE"] = df["DATE"].dt.to_period("M").dt.to_timestamp("M")
    return df.set_index("DATE").sort_index()
