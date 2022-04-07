import unittest

from names.rules_based import npc_gen


class TestNPC(unittest.TestCase):
    def setUp(self):
        self.lines = [
            "",
            "[S]=[M]",
            "[M]=Abdallah\n",
            "[mpre]=ivelli",
            "[F]=[fpre][fin][fsuf]",
            "[F']=Anastrianna\n",
        ]

    def tearDown(self):
        pass

    def test_female_humans_df(self):
        df = npc_gen.female_human_df()
        self.assertEqual(127, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[100].to_dict()
        expected = {"gender": "F", "name": "Thora", "source": "npc female human"}
        self.assertEqual(expected, actual)

    def test_male_humans_df(self):
        df = npc_gen.male_humans_df()
        self.assertEqual(357, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[100].to_dict()
        expected = {"gender": "M", "name": "Geirstein", "source": "npc male human"}
        self.assertEqual(expected, actual)

    def test_female_dwarf_df(self):
        df = npc_gen.female_dwarf_df()
        self.assertEqual(45 * 13, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[100].to_dict()
        expected = {"gender": "F", "name": "Dornal", "source": "npc female dwarf"}
        self.assertEqual(expected, actual)

    def test_male_dwarf_df(self):
        df = npc_gen.male_dwarf_df()
        self.assertEqual(45 * 15, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[100].to_dict()
        expected = {"gender": "M", "name": "Dkil", "source": "npc male dwarf"}
        self.assertEqual(expected, actual)

    def test_female_elf_df(self):
        df = npc_gen.female_elf_df()
        self.assertEqual(10 + 10 * 10 * 9, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[5].to_dict()
        expected = {"gender": "F", "name": "Lia", "source": "npc female elf"}
        self.assertEqual(expected, actual)

        actual = df.iloc[10].to_dict()
        expected = {"gender": "F", "name": "Anastrianna", "source": "npc female elf"}
        self.assertEqual(expected, actual)

        actual = df.iloc[10 + 501].to_dict()
        expected = {"gender": "F", "name": "Liquiqui", "source": "npc female elf"}
        self.assertEqual(expected, actual)

    def test_male_elf_df(self):
        df = npc_gen.male_elf_df()
        self.assertEqual(10 + 10 * 4 * 9, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[5].to_dict()
        expected = {"gender": "M", "name": "Ivellios", "source": "npc male elf"}
        self.assertEqual(expected, actual)

        actual = df.iloc[10].to_dict()
        expected = {"gender": "M", "name": "Aravil", "source": "npc male elf"}
        self.assertEqual(expected, actual)

        actual = df.iloc[10 + 301].to_dict()
        expected = {"gender": "M", "name": "Thamimo", "source": "npc male elf"}
        self.assertEqual(expected, actual)

    def test_female_elf_permuted_names(self):
        names = npc_gen.permuted_elf_names(False)
        self.assertEqual(10 * 10 * 9, len(names))
        self.assertEqual("Anastrianna", names[0])
        self.assertEqual("Liquiqui", names[501])

    def test_male_elf_permuted_names(self):
        names = npc_gen.permuted_elf_names(True)
        self.assertEqual(10 * 4 * 9, len(names))
        self.assertEqual("Aravil", names[0])
        self.assertEqual("Thamimo", names[301])

    def test_female_gnome_df(self):
        df = npc_gen.female_gnome_df()
        self.assertEqual(10, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[5].to_dict()
        expected = {"gender": "F", "name": "Loopmottin", "source": "npc female gnome"}
        self.assertEqual(expected, actual)

    def test_male_gnome_df(self):
        df = npc_gen.male_gnome_df()
        self.assertEqual(10, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[5].to_dict()
        expected = {"gender": "M", "name": "Jebeddo", "source": "npc male gnome"}
        self.assertEqual(expected, actual)

    def test_female_halfling_df(self):
        df = npc_gen.female_halfling_df()
        self.assertEqual(34, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[5].to_dict()
        expected = {"gender": "F", "name": "Lavinia", "source": "npc female halfling"}
        self.assertEqual(expected, actual)

    def test_male_halfling_df(self):
        df = npc_gen.male_halfling_df()
        self.assertEqual(20, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[5].to_dict()
        expected = {"gender": "M", "name": "Lyle", "source": "npc male halfling"}
        self.assertEqual(expected, actual)

    def test_count_tags(self):
        expected = [0, 2, 1, 1, 4, 1]
        actual = [npc_gen.count_tags(line) for line in self.lines]
        self.assertEqual(expected, actual)

    def test_find_tags(self):
        expected = [
            [],
            ["S", "M"],
            ["M"],
            ["mpre"],
            ["F", "fpre", "fin", "fsuf"],
            ["F'"],
        ]
        actual = [npc_gen.find_tags(line) for line in self.lines]
        self.assertEqual(expected, actual)

    def test_strip_tags(self):
        expected = ["", "", "Abdallah", "ivelli", "", "Anastrianna"]
        actual = [npc_gen.strip_tags(line) for line in self.lines]
        self.assertEqual(expected, actual)

    def test_is_tagged(self):
        self.assertTrue(npc_gen.is_tagged("foo", "[foo]=adfgzdfgzdfg"))
        self.assertFalse(npc_gen.is_tagged("foo", "[foo]=[bar]"))
        self.assertFalse(npc_gen.is_tagged("foo", "[bla]=[foo]"))
        self.assertFalse(npc_gen.is_tagged("foo", "[F]=adfgzdfgzdfg"))
        self.assertFalse(npc_gen.is_tagged("foo", ""))

    def test_is_male(self):
        self.assertTrue(npc_gen.is_male("[M]=adfgzdfgzdfg"))
        self.assertTrue(npc_gen.is_male("[M']=adfgzdfgzdfg"))
        self.assertFalse(npc_gen.is_male("[foo]=[bar]"))
        self.assertFalse(npc_gen.is_male("[bla]=[foo]"))
        self.assertFalse(npc_gen.is_male("[F]=adfgzdfgzdfg"))
        self.assertFalse(npc_gen.is_male(""))

    def test_is_female(self):
        self.assertTrue(npc_gen.is_female("[F]=adfgzdfgzdfg"))
        self.assertTrue(npc_gen.is_female("[F']=adfgzdfgzdfg"))
        self.assertFalse(npc_gen.is_female("[foo]=[bar]"))
        self.assertFalse(npc_gen.is_female("[bla]=[foo]"))
        self.assertFalse(npc_gen.is_female("[M]=adfgzdfgzdfg"))
        self.assertFalse(npc_gen.is_female(""))

    def test_female_elf_partnames(self):
        partnames = npc_gen.elf_part_names(False)
        self.assertEqual(3, len(partnames))
        pre = partnames["pre"]
        mid = partnames["in"]
        suf = partnames["suf"]
        self.assertEqual(10, len(pre))
        self.assertEqual(9, len(mid))
        self.assertEqual(9, len(suf))
        expected = {
            "pre": [
                "ana",
                "anti",
                "drusi",
                "felo",
                "iele",
                "li",
                "qilla",
                "sila",
                "vala",
                "xana",
            ],
            "in": ["stria", "nu", "si", "ni", "la", "qui", "qua", "nthe", "phi"],
            "suf": ["nna", "nua", "lia", "sial", "nia", "the", "qui", "nthe", "phia"],
        }
        self.assertEqual(expected, partnames)

    def test_male_elf_partnames(self):
        partnames = npc_gen.elf_part_names(True)
        self.assertEqual(3, len(partnames))
        pre = partnames["pre"]
        mid = partnames["in"]
        suf = partnames["suf"]
        self.assertEqual(10, len(pre))
        self.assertEqual(3, len(mid))
        self.assertEqual(9, len(suf))
        expected = {
            "pre": [
                "ara",
                "a",
                "enia",
                "he",
                "himo",
                "ivelli",
                "lauci",
                "quari",
                "thami",
                "thari",
            ],
            "in": ["v", "m", "l"],
            "suf": ["il", "ust", "is", "ian", "o", "os", "on", "or", "ol"],
        }
        self.assertEqual(expected, partnames)

    def test_female_dwarf_partnames(self):
        partnames = npc_gen.dwarf_part_names(False)
        self.assertEqual(2, len(partnames))
        pre = partnames["pre"]
        suf = partnames["suf"]
        self.assertEqual(45, len(pre))
        self.assertEqual(13, len(suf))
        expected = {
            "pre": [
                "bal",
                "belf",
                "bif",
                "bof",
                "bol",
                "bomb",
                "d",
                "dor",
                "dorf",
                "dur",
                "dwal",
                "gar",
                "fil",
                "gil",
                "gol",
                "gor",
                "kon",
                "kor",
                "kur",
                "mor",
                "na",
                "no",
                "nor",
                "o",
                "or",
                "thor",
                "thra",
                "thro",
                "tor",
                "whar",
                "ulf",
                "art",
                "aud",
                "dag",
                "gunn",
                "bar",
                "brott",
                "eb",
                "ein",
                "osk",
                "rur",
                "tak",
                "hl",
                "lift",
                "torgg",
            ],
            "suf": [
                "a",
                "as",
                "i",
                "ia",
                "if",
                "il",
                "is",
                "la",
                "hild",
                "nal",
                "loda",
                "rasa",
                "ga",
            ],
        }
        self.assertEqual(expected, partnames)

    def test_male_dwarf_partnames(self):
        partnames = npc_gen.dwarf_part_names(True)
        self.assertEqual(2, len(partnames))
        pre = partnames["pre"]
        suf = partnames["suf"]
        self.assertEqual(45, len(pre))
        self.assertEqual(15, len(suf))
        expected = {
            "pre": [
                "bal",
                "belf",
                "bif",
                "bof",
                "bol",
                "bomb",
                "d",
                "dor",
                "dorf",
                "dur",
                "dwal",
                "gar",
                "fil",
                "gil",
                "gol",
                "gor",
                "kon",
                "kor",
                "kur",
                "mor",
                "na",
                "no",
                "nor",
                "o",
                "or",
                "thor",
                "thra",
                "thro",
                "tor",
                "whar",
                "ulf",
                "art",
                "aud",
                "dag",
                "gunn",
                "bar",
                "brott",
                "eb",
                "ein",
                "osk",
                "rur",
                "tak",
                "hl",
                "lift",
                "torgg",
            ],
            "suf": [
                "ar",
                "ed",
                "ic",
                "in",
                "lum",
                "or",
                "to",
                "ur",
                "endd",
                "erk",
                "kil",
                "ik",
                "linn",
                "bon",
                "gar",
            ],
        }
        self.assertEqual(expected, partnames)
