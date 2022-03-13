import os
import json
import pickle

import tensorflow as tf

from names import directories, seq_model, vectorize


class CheckPoint:
    def __init__(self, base='.', checkpoint_dir="training_checkpoints", prefix="ckpt", reload=False):
        self.base = os.path.abspath(base)
        if reload:
            self.load_info()
        else:
            self.d = {'checkpoint_dir': checkpoint_dir, 'prefix': prefix}
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_prefix, save_weights_only=True
        )

    @property
    def checkpoint_dir(self):
        return directories.qualifyname(self.base, self.d['checkpoint_dir'])

    @property
    def checkpoint_prefix(self):
        return directories.qualifyname(self.checkpoint_dir, self.d['prefix']) + '_{epoch}'

    @property
    def info_path(self):
        return directories.qualifyname(self.base, 'info.json')

    @property
    def vocab_path(self):
        return directories.qualifyname(self.base, 'vocab.pkl')

    @property
    def model_path(self):
        return directories.qualifyname(self.base, 'seq_model')

    def create_dirs(self):
        if not os.path.isdir(self.base):
            os.mkdir(self.base)
            os.mkdir(self.checkpoint_dir)

    def save_info(self):
        self.create_dirs()
        with open(self.info_path, 'w') as f:
            json.dump(self.d, f)

    def load_info(self):
        with open(self.info_path) as f:
            self.d = json.load(f)

    def save(self, vocab, model):
        self.save_info()
        with open(self.vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        tf.saved_model.save(model, self.model_path)

    def load(self):
        self.load_info()
        with open(self.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        vectorator = vectorize.Vectorator(vocab)
        model = tf.saved_model.load(self.model_path)
        one_step_model = seq_model.OneStep(model, vectorator=vectorator)
        return vocab, vectorator, model, one_step_model
