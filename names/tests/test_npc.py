import unittest

from names.rules_based import npc_gen


class TestNPC(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_female_humans_df(self):
        df = npc_gen.female_human_df()
        self.assertEqual(127, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[100].to_dict()
        expected = {'gender': 'F', 'name': 'Thora', 'source': 'npc female human'}
        self.assertEqual(expected, actual)

    def test_male_humans_df(self):
        df = npc_gen.male_humans_df()
        self.assertEqual(357, df.name.count())
        expected = ["name", "gender", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[100].to_dict()
        expected = {'gender': 'M', 'name': 'Geirstein', 'source': 'npc male human'}
        self.assertEqual(expected, actual)
