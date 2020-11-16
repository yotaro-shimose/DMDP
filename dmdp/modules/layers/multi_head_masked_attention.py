import tensorflow as tf
import numpy as np
from dmdp.modules.layers.masked_attention import MaskedAttention


class MultiHeadMaskedAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, n_heads, weight_balancer=0.01):
        super().__init__()
        if (d_model % n_heads) != 0:
            raise Exception('割り切れる数字入れてね！！')
        self.d_model = d_model
        self.d_key = d_key
        self.n_heads = n_heads
        self.sat_list = [MaskedAttention(int(d_model/n_heads), d_key, weight_balancer)
                         for i in range(n_heads)]
        self.weight_balancer = weight_balancer

    def build(self, input_shape):
        shape = (int(self.d_model/self.n_heads), self.d_model)
        initializer = tf.random_uniform_initializer(
            -np.sqrt(6 / (shape[0] + shape[1])) * self.weight_balancer,
            np.sqrt(6 / (shape[0] + shape[1])) * self.weight_balancer
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
