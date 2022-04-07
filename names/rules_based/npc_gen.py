import pandas as pd

from names import directories

def npc_filename(filename):
    path = directories.data("npc")
    return directories.qualifyname(path, filename)

def humans():
    with open(npc_filename("human_names.txt")) as f:
        return [line.strip() for line in f if len(line.strip())]

def female_humans():
    hn = humans()
    return [line[4:] for line in hn if line[:4] == "[F]="]

def male_humans():
    hn = humans()
    return [line[4:] for line in hn if line[:4] == "[M]="]


def female_human_df():
    df = pd.DataFrame()
    df['name'] = female_humans()
    df['gender'] = "F"
    df['source'] = "npc female human"
    return df

def male_humans_df():
    df = pd.DataFrame()
    df['name'] = male_humans()
    df['gender'] = "M"
    df['source'] = "npc male human"
    return df
