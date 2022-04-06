import unittest

from names.rules_based import from_file


class TestDataFromFile(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_jrpg_df(self):
        df = from_file.jrpg()
        self.assertEqual(6455, df.name.count())
        expected = ["name", "source"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[1000].to_dict()
        expected = {"name": "Cid Highwind", "source": "jrpg names"}
        self.assertEqual(expected, actual)
