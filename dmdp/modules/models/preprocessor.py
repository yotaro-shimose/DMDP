import tensorflow as tf


class Preprocessor(tf.keras.models.Model):

    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.n_heads = n_heads
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        '''
        inputs === [H, times, status, masks]
        outputs === [H, h_c, masks]
        h_c === [h_g, h_0, h_t]
        '''
        # Parse inputs.
        # B, N, D
        H = inputs[0]
        # B
        times = inputs[1]
        # B, 3
        status = inputs[2]
        # B
        masks = inputs[3]
        # B
        currents = status[:, 0]

        # pick up current nodes' embeddings from H
        # B, N, D
        indices = tf.tile(tf.expand_dims(tf.one_hot(
            currents, depth=H.shape[1], dtype=tf.int32), -1), [1, 1, H.shape[-1]])
        # B, D
        h_p = tf.reduce_sum(H * tf.cast(indices, tf.float32), axis=1)

        # calculate h_g
        # B, D
        h_g = tf.reduce_mean(H, axis=1)
        # B, D + D + 1 + 3
        h_c = tf.concat(
            [h_g, h_p, tf.expand_dims(times, -1), tf.cast(status, tf.float32)], axis=-1)

        # Point Wise Dense
        # B, D
        h_c = self.dense(h_c)

        # return created inputs for decoder.
        return [H, tf.expand_dims(h_c, 1), masks]
