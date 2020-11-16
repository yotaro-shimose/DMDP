import tensorflow as tf
from dmdp.modules.functions import masked_softmax
from dmdp.modules.layers.multi_head_self_attention import MultiHeadSelfAttention
from dmdp.modules.models.preprocessor import Preprocessor


class PolicyDecoder(tf.keras.models.Model):
    def __init__(self, d_model, d_key, n_heads, th_range, weight_balancer=0.01):
        super().__init__()
        if (d_model % n_heads) != 0:
            raise ValueError('d_model must be multiple of n_heads!')
        self.d_model = d_model
        self.d_key = d_key
        self.attention = MultiHeadSelfAttention(
            d_model, d_key, n_heads, weight_balancer)
        self.th_range = th_range
        self.preprocessor = Preprocessor(d_model, d_key, n_heads)
        self.weight_balancer = weight_balancer

    def build(self, input_shape):

        initializer = tf.random_uniform_initializer(
            -tf.sqrt(6/(self.d_model + self.d_key)) * self.weight_balancer,
            tf.sqrt(6/(self.d_model + self.d_key)) * self.weight_balancer
        )

        self.wq = self.add_weight(name="wq", shape=(self.d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

        self.wk = self.add_weight(name="wk", shape=(self.d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

    def calc_policy(self, Q, K, mask):
        divide_const = tf.sqrt(tf.constant(K.shape[-1], dtype=tf.float32))
        QK = tf.matmul(Q, K, transpose_b=True) / divide_const
        # mask is tensor of shape (batch_size, n_nodes) by default.
        # but it must be tensor of shape (batch_size, 1, n_nodes).
        batch_size = Q.shape[0]
        n_query = Q.shape[1]  # always one in ordinary use
        n_nodes = K.shape[1]
        mask = tf.reshape(mask, (batch_size, n_query, n_nodes))
        policy = masked_softmax(
            self.th_range * tf.keras.activations.tanh(QK), mask)
        # now policy is tensor of shape(batch_size, 1, n_nodes) which must be turned into tensor of
        # shape(batch_size, n_nodes)
        return tf.reshape(policy, (batch_size, n_nodes))

    def call(self, inputs):
        """

        Args:
            inputs (list): [graph, time, status, mask]

        Returns:
            tf.Tensor: B, N
        """
        H, h_c, mask = self.preprocessor(inputs)
        query = self.attention([H, h_c, mask])
        return self.calc_policy(tf.matmul(query, self.wq), tf.matmul(
            H, self.wk), mask)
