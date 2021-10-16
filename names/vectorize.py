import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def extract_vocab(text):
    return sorted(set(text))


def unicode_split(text):
    return tf.strings.unicode_split(text, input_encoding='UTF-8')


class Vectorator:
    def __init__(self, vocab):

        self.ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(vocab), mask_token=None
        )

        self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(),
            invert=True,
            mask_token=None,
        )

    def text_from_ids(self, ids):
        text = tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)
        return [snippet.decode("utf-8") for snippet in text.numpy()]
