import tensorflow as tf

from dmdp.modules.models.residual import ResidualLayerNorm
from dmdp.modules.layers.multi_head_self_attention import MultiHeadSelfAttention
import numpy as np


class Encoder(tf.keras.models.Model):
    def __init__(self, d_model, d_key, n_heads, depth, weight_balancer=0.01):
        super().__init__()
        self.weight_balancer = weight_balancer
        self.d_model = d_model
        self.attention_block_list = [ResidualLayerNorm(
            MultiHeadSelfAttention(d_model, d_key, n_heads, weight_balancer)) for _ in range(depth)]
        self.dence_block_list = [ResidualLayerNorm(
            tf.keras.layers.Dense(d_model, activation='relu')) for _ in range(depth)]
        self.final_layer_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):

        initializer = tf.random_uniform_initializer(
            -np.sqrt(6 / (input_shape[-1] + self.d_model)
                     ) * self.weight_balancer,
            np.sqrt(6 / (input_shape[-1] + self.d_model)
                    ) * self.weight_balancer
        )
        self.w = self.add_weight(name="w", shape=(input_shape[-1], self.d_model),
                                 initializer=initializer,
                                 trainable=True)

    def call(self, x):
        '''
        input === (BATCH_SIZE, n_nodes, d_feature(2))
        output === (BATCH_SIZE, n_nodes, d_model)
        '''
        x = tf.tensordot(x, self.w, axes=[2, 0])
        for attention_block, dence_block in zip(self.attention_block_list, self.dence_block_list):
            x = tf.add(attention_block(x), x)
            x = tf.add(dence_block(x), x)
        x = self.final_layer_norm(x)
        return x
