#!/usr/bin/env python
import unittest

from names.tests.test_census import TestCensus
from names.tests.test_checkpoint import TestCheck
from names.tests.test_data_from_file import TestDataFromFile
from names.tests.test_directories import TestDirectories
from names.tests.test_example import TestExample
from names.tests.test_npc import TestNPC
from names.tests.test_raw_data import TestRawData
from names.tests.test_seq_model import TestSeqModel
from names.tests.test_vectorize import TestVectorize


class CountSuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d: %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite():
    s = CountSuite()

    s.add(TestCensus)
    s.add(TestCheck)
    s.add(TestDataFromFile)
    s.add(TestDirectories)
    s.add(TestExample)
    s.add(TestNPC)
    s.add(TestRawData)
    s.add(TestSeqModel)
    s.add(TestVectorize)

    return s.s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
