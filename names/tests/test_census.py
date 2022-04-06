import unittest

from names.rules_based import census


class TestCensus(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_census_df(self):
        df = census.census(1880, 1882)
        self.assertEqual(6061, df.name.count())
        expected = ["name", "gender", "count", "year"]
        actual = list(df.columns)
        self.assertEqual(expected, actual)
        actual = df.iloc[1000].to_dict()
        expected = {"count": 305, "gender": "M", "name": "Charley", "year": 1880}
        self.assertEqual(expected, actual)
