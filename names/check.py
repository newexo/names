import os
import json
import pickle

import tensorflow as tf

from names import directories, seq_model, vectorize


class CheckPoint:
    def __init__(self, base='.', checkpoint_dir="training_checkpoints", prefix="ckpt", callback=False):
        self.base = os.path.abspath(base)
        if os.path.exists(self.info_path):
            self.load_info()
        else:
            self.d = {'checkpoint_dir': checkpoint_dir, 'prefix': prefix, "hypers": None}
        self.create_dirs()
        if callback:
            self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_prefix, save_weights_only=True
            )
        else:
            self.checkpoint_callback = None

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

    @property
    def exists(self):
        return os.path.isdir(self.base)

    def create_dirs(self):
        if not os.path.isdir(self.base):
            os.mkdir(self.base)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def save_info(self):
        self.create_dirs()
        with open(self.info_path, 'w') as f:
            json.dump(self.d, f)

    def load_info(self):
        with open(self.info_path) as f:
            self.d = json.load(f)

    def save(self, model: seq_model.SeqModel):
        self.d["hypers"] = model.hypers.to_dict()
        self.save_info()
        with open(self.vocab_path, "wb") as f:
            pickle.dump(model.vocab, f)
        model.save_weights(self.model_path)

    def load(self):
        self.load_info()
        with open(self.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        model = seq_model.SeqModel(vocab, seq_model.Hypers.restore(self.d["hypers"]))
        model.load_weights(self.model_path)
        model.compile_model()
        return model
