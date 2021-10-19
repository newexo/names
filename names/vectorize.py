import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def extract_vocab(text):
    return sorted(set(text))


def unicode_split(text):
    return tf.strings.unicode_split(text, input_encoding="UTF-8")


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


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

    @property
    def vocab_size(self):
        return self.chars_from_ids.vocabulary_size()


class DataSet(Vectorator):
    def __init__(
        self, text, vocab=None, seq_length=100, batch_size=64, buffer_size=10000
    ):
        if vocab is None:
            vocab = extract_vocab(text)
        Vectorator.__init__(self, vocab)

        all_ids = self.ids_from_chars(unicode_split(text))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        self.examples_per_epoch = len(text) // (seq_length + 1)
        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
        dataset = sequences.map(split_input_target)
        self.dataset = (
            dataset.shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
