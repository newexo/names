import unittest
import tensorflow as tf
import numpy as np
from numpy.linalg import norm

from . import fixture

from names import vectorize

class TestVectorize(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.Fixture()
        self.example_text = ['abcdefg', 'xyz']
        self.vectorator = vectorize.Vectorator(self.fixture.vocab)

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

    def test_ids_from_chars(self):
        chars = vectorize.unicode_split(self.example_text)
        ids = self.vectorator.ids_from_chars(chars)
        actual = ids.numpy()
        expected = [np.array([45, 46, 47, 48, 49, 50, 51], dtype=np.int64),
            np.array([68, 69, 70], dtype=np.int64)]
        for i in range(2):
            self.assertAlmostEqual(0, norm(actual[i] - expected[i]))

    def test_text_from_ids(self):
        chars = vectorize.unicode_split(self.example_text)
        ids = self.vectorator.ids_from_chars(chars)
        actual = self.vectorator.text_from_ids(ids)
        expected = ['abcdefg', 'xyz']
        self.assertEqual(expected, actual)
