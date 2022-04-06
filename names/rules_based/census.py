import pandas as pd
from names import directories


def yob_filename(year):
    name = "yob{}.txt".format(year)
    return directories.qualifyname(directories.data("census"), name)


def yob(year):
    df = pd.read_csv(yob_filename(year), names=["name", "gender", "count"])
    df["year"] = year
    df["source"] = "US census"
    return df


def census(start=1880, end=2020):
    frames = [yob(i) for i in range(start, end + 1)]
    return pd.concat(frames)
