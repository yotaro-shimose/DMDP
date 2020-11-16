import tensorflow as tf


class ResidualBatchNorm(tf.keras.models.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x1 = tf.add(self.layer(x), x)
        return tf.add(self.batch_normalization(x1), x1)


class ResidualLayerNorm(tf.keras.models.Model):
    """ResidualLayerNorm is a residual block with an arbitrary layer followed by layer normalization.
    according to https://arxiv.org/pdf/2002.04745.pdf, transformer should have
    layer norm inside the residual block, or the output of each layer explodes as it goes deeper.
    and layer norm is prefered to batch norm in recent transformers' architectures.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layer_normalization = tf.keras.layers.LayerNormalization()

    def call(self, x):
        residual = tf.identity(x)
        x = self.layer_normalization(x)
        return tf.add(self.layer(x), residual)
