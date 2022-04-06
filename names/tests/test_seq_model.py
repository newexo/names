import unittest

from names import seq_model


class TestSeqModel(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_save_hypers(self):
        h = seq_model.Hypers(257, [21, 42])
        actual = h.to_dict()
        expected = {'embedding_dim': 257, 'rnn_units': [21, 42]}
        self.assertEqual(expected, actual)

    def test_restore_hypers(self):
        d = {'embedding_dim': 257, 'rnn_units': [21, 42], 'vocab_size': 103}
        h = seq_model.Hypers.restore(d)
        self.assertEqual(257, h.embedding_dim)
        self.assertEqual([21, 42], h.rnn_units)
