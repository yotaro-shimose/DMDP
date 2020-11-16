import tensorflow as tf
from dmdp.modules.functions import masked_softmax


class MaskedAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, weight_balancer):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.weight_balancer = weight_balancer

    def build(self, input_shape):
        self.wq = self.add_weight(
            name="wq",
            shape=(input_shape[1][-1], self.d_key),
            initializer=tf.random_uniform_initializer(
                -tf.sqrt(6 / (input_shape[1][-1] +
                              self.d_key)) * self.weight_balancer,
                tf.sqrt(6 / (input_shape[1][-1] +
                             self.d_key)) * self.weight_balancer
            ),
            trainable=True
        )
        self.wk = self.add_weight(
            name="wk",
            shape=(input_shape[0][-1], self.d_key),
            initializer=tf.random_uniform_initializer(
                -tf.sqrt(6/(input_shape[0][-1] + self.d_key)
                         ) * self.weight_balancer,
                tf.sqrt(6/(input_shape[0][-1] + self.d_key)
                        ) * self.weight_balancer
            ),
            trainable=True
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(input_shape[0][-1], self.d_model),
            initializer=tf.random_uniform_initializer(
                -tf.sqrt(6/(input_shape[0][-1] +
                            self.d_model)) * self.weight_balancer,
                tf.sqrt(6/(input_shape[0][-1]+self.d_model)
                        ) * self.weight_balancer
            ),
            trainable=True
        )

    def masked_attention(self, Q, K, V, mask):
        divide_const = tf.sqrt(tf.constant(K.shape[-1], dtype=tf.float32))
        QK = tf.matmul(Q, K, transpose_b=True) / divide_const
        # mask is tensor of shape (batch_size, n_nodes) by default.
        # but it must be tensor of shape (batch_size, 1, n_nodes).
        batch_size = Q.shape[0]
        n_nodes = V.shape[1]
        mask = tf.reshape(mask, (batch_size, 1, n_nodes))
        weights = masked_softmax(QK, mask)
        return tf.matmul(weights, V)

    def call(self, inputs):
        return self.masked_attention(tf.matmul(inputs[1], self.wq), tf.matmul(
            inputs[0], self.wk), tf.matmul(inputs[0], self.wv), inputs[2])
