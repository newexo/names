import unittest
import hashlib

from names.raw_data import join_raw
from . import fixture


class TestRawData(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.Fixture()

    def tearDown(self):
        pass

    def test_load_unicode_without_error(self):
        # first check a few names
        expected = "Adray Lasbard"
        actual = self.fixture.raw[0]
        self.assertEqual(expected, actual)

        expected = "Albel Nox"
        actual = self.fixture.raw[1]
        self.assertEqual(expected, actual)

        expected = "Cliff Fittir"
        actual = self.fixture.raw[2]
        self.assertEqual(expected, actual)

        # so far so good, but there were invalid bytes later in the file
        # compute the hash and check the whole thing
        actual = hashlib.sha1("\n".join(self.fixture.raw).encode("utf8")).hexdigest()
        expected = "adfc7087bd676d825a933cd781b5ee90aedb8f00"
        self.assertEqual(expected, actual)

    def test_ljust_lines(self):
        # first check a few hundred characters
        expected = "Adray Lasbard       \nAlbel Nox           \nCliff Fittir      "
        actual = self.fixture.text[:60]
        self.assertEqual(expected, actual)

        # so far so good, but there were invalid bytes later in the file
        # compute the hash and check the whole thing
        actual = hashlib.sha1(self.fixture.text.encode("utf8")).hexdigest()
        expected = "c88737d1e71350a525b699b0c07d3118bcac6303"
        self.assertEqual(expected, actual)

    def test_lines_with_overflow(self):
        lines = ["", "12", "12345", "123"]
        expected = "    \n12  \n1234\n123 "
        actual = join_raw(lines, ljust=4)
        self.assertEqual(expected, actual)
