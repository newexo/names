import re
import pandas as pd

from names import directories


def npc_filename(filename):
    path = directories.data("npc")
    return directories.qualifyname(path, filename)


tag_regx = re.compile(r"\[([^\]]+)\]=?")


def find_tags(line):
    return tag_regx.findall(line)


def strip_tags(line):
    return tag_regx.sub("", line.strip())


def is_tagged(tag, line):
    tags = find_tags(line)
    return len(tags) == 1 and tags[0] == tag


def is_female(line):
    return is_tagged("F", line) or is_tagged("F'", line)


def is_male(line):
    return is_tagged("M", line) or is_tagged("M'", line)


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


def female_dwarf_df():
    return gendered_df(
        "dwarf", False, lambda race, using_male: permuted_dwarf_names(using_male)
    )


def male_dwarf_df():
    return gendered_df(
        "dwarf", True, lambda race, using_male: permuted_dwarf_names(using_male)
    )


def name_parts(filename, part_names):
    return [
        filter_strip_file(
            filename, lambda line: is_tagged(part_name, line), should_capitalize=False
        )
        for part_name in part_names
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
    for pre in parts["pre"]:
        for mid in parts["in"]:
            for suf in parts["suf"]:
                name = pre + mid + suf
                name = name.capitalize()
                names.append(name)
    return names


def dwarf_part_names(is_male):
    letter = "m" if is_male else "f"
    part_names = ["pre", "suf"]
    gendered_part_names = ["pre", letter + "suf"]
    parts = name_parts("dwarf_names.txt", gendered_part_names)
    return {part_name: part for part_name, part in zip(part_names, parts)}


def permuted_dwarf_names(is_male):
    parts = dwarf_part_names(is_male)
    names = []
    for pre in parts["pre"]:
        for suf in parts["suf"]:
            name = pre + suf
            name = name.capitalize()
            names.append(name)
    return names
