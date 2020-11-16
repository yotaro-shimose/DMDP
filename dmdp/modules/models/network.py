from tensorflow.python.keras.mixed_precision.experimental.policy import Policy
from dmdp.modules.models.encoder import Encoder
from dmdp.modules.models.decoder import PolicyDecoder

import tensorflow as tf


class GraphAttentionNetwork(tf.keras.models.Model):
    def __init__(self, d_model, d_key, n_heads, depth, th_range, weight_balancer=0.01):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model,
            d_key=d_key,
            n_heads=n_heads,
            depth=depth,
            weight_balancer=weight_balancer
        )

        self.decoder = PolicyDecoder(
            d_model=d_model,
            d_key=d_key,
            n_heads=n_heads,
            th_range=th_range,
            weight_balancer=weight_balancer
        )

    def call(self, inputs):
        graph = inputs[0]
        H = self.encoder(graph)
        policy = self.decoder([H] + inputs[1:])
        return policy
