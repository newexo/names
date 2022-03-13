from names import directories, raw_data, vectorize


class Fixture:
    def __init__(self):
        self.raw = raw_data.load_raw(directories.data("names.txt"))
        self.text = raw_data.join_raw(self.raw, ljust=20)
        self.vocab = vectorize.extract_vocab(self.text)
