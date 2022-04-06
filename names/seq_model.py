import tensorflow as tf

from names import vectorize


class Hypers:
    def __init__(self, embedding_dim, rnn_units):
        self.embedding_dim = embedding_dim
        if not hasattr(rnn_units, "__iter__"):
            rnn_units = [rnn_units]
        self.rnn_units = rnn_units

    @staticmethod
    def restore(d):
        return Hypers(d["embedding_dim"], d["rnn_units"])

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "rnn_units": self.rnn_units,
        }


class SeqModel(tf.keras.Model):
    def __init__(self, vocab, hypers: Hypers=None, embedding_dim=None, rnn_units=None):
        super().__init__(self)
        self.vocab = vocab
        self.vectorator = vectorize.Vectorator(vocab)
        if hypers is None:
            hypers = Hypers(embedding_dim, rnn_units)
        self.hypers = hypers
        self.embedding = tf.keras.layers.Embedding(
            self.vectorator.vocab_size, hypers.embedding_dim
        )
        self.grus = [
            tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
            for rnn_units in hypers.rnn_units
        ]
        self.dense = tf.keras.layers.Dense(self.vectorator.vocab_size)
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = [None] * len(self.grus)
        new_states = []
        for states, gru in zip(states, self.grus):
            if states is None:
                states = gru.get_initial_state(x)
            x, new_state = gru(x, initial_state=states, training=training)
            new_states.append(new_state)
        x = self.dense(x, training=training)

        if return_state:
            return x, new_states
        else:
            return x

    def compile_model(self):
        self.compile(optimizer="adam", loss=self.loss)

    def fit_with_chk(self, ds, checkpoint, epochs, initial_epoch=0):
        if checkpoint.checkpoint_callback is None:
            return self.fit(
                ds.dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
            )
        else:
            return self.fit(
                ds.dataset,
                epochs=epochs,
                callbacks=[checkpoint.checkpoint_callback],
                initial_epoch=initial_epoch,
            )


class OneStep(tf.keras.Model):
    def __init__(
        self,
        model: SeqModel,
        temperature=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.model = model

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(self.ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @property
    def chars_from_ids(self):
        return self.model.vectorator.chars_from_ids

    @property
    def ids_from_chars(self):
        return self.model.vectorator.ids_from_chars


    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

    def generate(self, first_chars, n):
        states = None
        next_char = tf.constant(first_chars)
        result = [next_char]

        for _ in range(n):
            next_char, states = self.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        return [line.decode("utf-8") for line in result.numpy()]
