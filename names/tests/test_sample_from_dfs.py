import unittest
import numpy as np

from names.rules_based import npc_gen
from names.raw_data import row2sample, sample_from_dfs


class TestSampleFromDfs(unittest.TestCase):
    def setUp(self):
        self.human = npc_gen.female_human_df()
        self.gnome = npc_gen.female_gnome_df()
        self.elf = npc_gen.female_elf_df()

    def tearDown(self):
        pass

    def test_row2sample(self):
        row = self.human.iloc[1]
        expected = "{'gender': 'F', 'source': 'npc female human'}\nAgatha"
        actual = row2sample(row)
        self.assertEqual(expected, actual)

    def test_sample_from_dfs(self):
        expeceted = [
            "{'gender': 'F', 'source': 'npc female human'}\nCatherine",
            "{'gender': 'F', 'source': 'npc female human'}\nThyra",
            "{'gender': 'F', 'source': 'npc female gnome'}\nRoywyn",
            "{'gender': 'F', 'source': 'npc female gnome'}\nBimpnottin",
            "{'gender': 'F', 'source': 'npc female elf'}\nIelesiqui",
            "{'gender': 'F', 'source': 'npc female elf'}\nAnathe",
        ]
        actual = sample_from_dfs(
            [self.human, self.gnome, self.elf], k=2, r=np.random.RandomState(42)
        )
        self.assertEqual(expeceted, actual)
