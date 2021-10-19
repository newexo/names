import unittest
import hashlib

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
        expected = "3cb92c5d0c260d010ae31b9a8031a50ce6bb8a4c"
        self.assertEqual(expected, actual)
