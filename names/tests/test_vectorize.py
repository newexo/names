import unittest

from . import fixture


class TestVectorize(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.Fixture()

    def tearDown(self):
        pass

    def test_vocab(self):
        expected = [
            "\n",
            " ",
            "&",
            "'",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "5",
            "6",
            "7",
            "8",
            "9",
            "?",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "Ó",
            "á",
            "é",
            "í",
            "ó",
            "ö",
            "ā",
            "ī",
        ]
        actual = self.fixture.vocab
        self.assertEqual(expected, actual)

    # def test_foo(self):
    #     self.fail("incomplete")
