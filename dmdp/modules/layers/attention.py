import tensorflow as tf
from dmdp.modules.functions import attention


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, weight_balancer=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.weight_balancer = weight_balancer

    def build(self, input_shape):
        self.wq = self.add_weight(
            name="wq",
            shape=(input_shape[-1], self.d_key),
            initializer=tf.random_uniform_initializer(-tf.sqrt(
                6/(input_shape[-1] + self.d_key)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[-1] + self.d_key)) * self.weight_balancer),
            trainable=True
        )
        self.wk = self.add_weight(
            name="wq",
            shape=(input_shape[-1], self.d_key),
            initializer=tf.random_uniform_initializer(-tf.sqrt(
                6/(input_shape[-1] + self.d_key)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[-1] + self.d_key)) * self.weight_balancer),
            trainable=True
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(input_shape[-1], self.d_model),
            initializer=tf.random_uniform_initializer(-tf.sqrt(
                6/(input_shape[-1] + self.d_model)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[-1] + self.d_model)) * self.weight_balancer),
            trainable=True
        )

    def call(self, x):
        return attention(tf.matmul(x, self.wq), tf.matmul(
            x, self.wk), tf.matmul(x, self.wv))


class QueryAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, weight_balancer=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.weight_balancer = weight_balancer

    def build(self, input_shape):
        self.wq = self.add_weight(
            name="wq",
            shape=(input_shape[-1], self.d_key),
            initializer=tf.random_uniform_initializer(-tf.sqrt(
                6/(input_shape[-1] + self.d_key)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[-1] + self.d_key)) * self.weight_balancer),
            trainable=True
        )
        self.wk = self.add_weight(
            name="wq",
            shape=(input_shape[-1], self.d_key),
            initializer=tf.random_uniform_initializer(-tf.sqrt(
                6/(input_shape[-1] + self.d_key)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[-1] + self.d_key)) * self.weight_balancer),
            trainable=True
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(input_shape[-1], self.d_model),
            initializer=tf.random_uniform_initializer(-tf.sqrt(
                6/(input_shape[-1] + self.d_model)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[-1] + self.d_model)) * self.weight_balancer),
            trainable=True
        )

    def call(self, inputs):
        H, h_c = inputs
        # B, 1, D
        Q = tf.matmul(h_c, self.wq)
        K = tf.matmul(H, self.wk)
        V = tf.matmul(H, self.wv)
        tf.assert_rank(Q, 3)
        # B, 1, D
        return attention(Q, K, V)
