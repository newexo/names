import pandas as pd

from names import directories


def from_file(filename, source):
    df = pd.read_table(filename, names=["name"])
    df.drop_duplicates()
    df["source"] = source
    return df


def jrpg():
    return from_file(directories.data("names.txt"), "jrpg names")
