import re
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


tag_regx = re.compile(r"\[([^\]]+)\]=?")


def count_tags(line):
    return len(tag_regx.findall(line))


def find_tags(line):
    return tag_regx.findall(line)


def strip_tags(line):
    return tag_regx.sub("", line.strip())


def is_tagged(tag, line):
    tags = find_tags(line)
    return len(tags) == 1 and tags[0] == tag


def find_tagged(tag, f):
    return [strip_tags(line) for line in f if is_tagged(tag, line)]


def is_female(line):
    return is_tagged("F", line) or is_tagged("F'", line)


def find_females(f):
    return [strip_tags(line) for line in f if is_female(line)]


def is_male(line):
    return is_tagged("M", line) or is_tagged("M'", line)


def find_males(f):
    return [strip_tags(line) for line in f if is_female(line)]


def filter_strip_file(filename, pred, should_capitalize=True):
    def alter_line(line):
        line = strip_tags(line)
        if should_capitalize:
            return line.capitalize()
        else:
            return line

    with open(npc_filename(filename)) as f:
        return [alter_line(line) for line in f if pred(line)]


def gendered_from_file(race, using_male):
    is_gendered = is_male if using_male else is_female
    filename = "{}_names.txt".format(race)
    return filter_strip_file(filename, is_gendered)


def gendered_df(race, using_male, make_names=gendered_from_file):
    word = "male" if using_male else "female"
    letter = "M" if using_male else "F"
    source = "npc {} {}".format(word, race)
    df = pd.DataFrame()
    df["name"] = make_names(race, using_male)
    df["gender"] = letter
    df["source"] = source
    return df


def female_human_df():
    return gendered_df("human", False)


def male_humans_df():
    return gendered_df("human", True)


def female_gnome_df():
    return gendered_df("gnome", False)


def male_gnome_df():
    return gendered_df("gnome", True)


def female_halfling_df():
    return gendered_df("halfling", False)


def male_halfling_df():
    return gendered_df("halfling", True)

def list_elf_names(race, using_male):
    found = gendered_from_file(race, using_male)
    permuted = permuted_elf_names(using_male)
    return found + permuted

def female_elf_df():
   return gendered_df("elf", False, list_elf_names)


def male_elf_df():
   return gendered_df("elf", True, list_elf_names)


def name_parts(filename, part_names):
    return [filter_strip_file(
            filename, lambda line: is_tagged(part_name, line), should_capitalize=False
        ) for part_name in part_names
        ]


def elf_part_names(is_male):
    letter = "m" if is_male else "f"
    part_names = ["pre", "in", "suf"]
    gendered_part_names = [letter + part_name for part_name in part_names]
    parts = name_parts("elf_names.txt", gendered_part_names)
    return {part_name: part for part_name, part in zip(part_names, parts)}


def permuted_elf_names(is_male):
    parts = elf_part_names(is_male)
    parts["in"].append("")
    names = []
    for pre in parts['pre']:
        for mid in parts['in']:
            for suf in parts['suf']:
                name = pre + mid + suf
                name = name.capitalize()
                names.append(name)
    return names
