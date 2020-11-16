import tensorflow as tf
from dmdp.modules.layers.attention import SelfAttention


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, n_heads, weight_balancer=0.01):
        super().__init__()
        if (d_model % n_heads) != 0:
            raise Exception('d_model must be multiple of n_heads!')
        self.d_model = d_model
        self.d_key = d_key
        self.n_heads = n_heads
        self.sat_list = [SelfAttention(int(d_model/n_heads), d_key, weight_balancer)
                         for i in range(n_heads)]
        self.weight_balancer = weight_balancer
        self.array = tf.TensorArray(dtype=tf.float32)

    def build(self, input_shape):
        shape = (int(self.d_model/self.n_heads), self.d_model)
        initializer = tf.random_uniform_initializer(
            -tf.sqrt(6 / (shape[0] + shape[1])) * self.weight_balancer,
            tf.sqrt(6 / (shape[0] + shape[1])) * self.weight_balancer
        )
        self.wo_list = [self.add_weight(
            name=f"wo_{i}",
            shape=shape,
            initializer=initializer,
            trainable=True
        ) for i in range(self.n_heads)]

    def call(self, inputs):
        return tf.reduce_sum([tf.matmul(sat(inputs), wo) for sat, wo
                              in zip(self.sat_list, self.wo_list)], axis=0)
