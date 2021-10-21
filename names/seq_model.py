import tensorflow as tf
import os


class SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

    def compile_model(self):
        self.compile(optimizer="adam", loss=self.loss)

    def fit_with_chk(self, ds, checkpoint, epochs, initial_epoch=0):
        return self.fit(
            ds.dataset,
            epochs=epochs,
            callbacks=[checkpoint.checkpoint_callback],
            initial_epoch=initial_epoch,
        )


class CheckPoint:
    def __init__(self, checkpoint_dir="./training_checkpoints", prefix="ckpt_{epoch}"):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, prefix)

        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        )


class OneStep(tf.keras.Model):
    def __init__(self, model, vectorator=None, chars_from_ids=None, ids_from_chars=None, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        if vectorator is not None:
            chars_from_ids = vectorator.chars_from_ids
            ids_from_chars = vectorator.ids_from_chars
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

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
