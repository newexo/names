import numpy as np


def load_raw(filename):
    with open(filename, encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def row2sample(row):
    row = row.to_dict()
    name = row["name"]
    del row["name"]
    return "{}\n{}".format(row, name)


def sample_from_dfs(dfs, k, r=None):
    samples = []
    if r is None:
        r = np.random.RandomState()
    for df in dfs:
        n = df.name.count()
        for i in r.choice(n, min(k, n), replace=False):
            samples.append(row2sample(df.iloc[i]))
    return samples


def join_raw(lines, ljust=100):
    return "\n".join([name.ljust(ljust)[:ljust] for name in lines])
