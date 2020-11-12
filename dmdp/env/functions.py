import tensorflow as tf

# TODO: arguments check


def int_and(x: tf.Tensor, y: tf.Tensor):
    return x * y


def int_or(x: tf.Tensor, y: tf.Tensor):
    return tf.cast(tf.cast(x + y, tf.bool), tf.int32)


def int_not(x: tf.Tensor):
    return (-1) * x + 1


def int_xor(x: tf.Tensor, y: tf.Tensor):
    return tf.math.mod(x + y, 2)
