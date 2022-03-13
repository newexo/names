import unittest
import os

from names import directories


class TestDirectories(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_directories_exist(self):
        self.assertTrue(os.path.isdir(directories.base()))
        self.assertTrue(os.path.isdir(directories.code()))
        self.assertTrue(os.path.isdir(directories.tests()))
        self.assertTrue(os.path.isdir(directories.test_data()))

    def test_filenames(self):
        self.assertTrue(os.path.exists(directories.base("README.md")))
        self.assertTrue(os.path.exists(directories.code("__init__.py")))
        self.assertTrue(os.path.exists(directories.tests("__init__.py")))
        self.assertTrue(os.path.exists(directories.test_data("README.md")))
