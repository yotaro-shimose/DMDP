import tensorflow as tf


def attention(Q, K, V):
    divide_const = tf.sqrt(tf.cast(tf.constant(K.shape[-1]), tf.float32))
    return tf.matmul(tf.nn.softmax(tf.divide(tf.matmul(Q, K, transpose_b=True), divide_const)), V)


def masked_cross_entropy_from_Q(Q, Q_target, mask):
    p_target = masked_softmax(Q_target, mask)
    p = masked_softmax(Q, mask)
    return -tf.math.reduce_mean(tf.keras.layers.dot([p_target, masked_log(p, mask)], axes=1))


def masked_softmax(tensor, mask):
    tensor = tensor
    exps = tf.math.exp(tensor) * (1 - tf.cast(mask, tf.float32))
    softmax = exps / tf.math.reduce_sum(exps, -1, keepdims=True)
    return softmax


def masked_log(tensor, mask):
    float_mask = tf.cast(mask, tf.float32)
    log = tf.math.log((1 - float_mask) * tensor + float_mask * 1)
    return log


# compute mask for trajectory with shape(batch_size, node_size)
@tf.function
def create_mask(trajectory):
    def _create_mask(trajectory):
        tf_range = tf.range(tf.size(trajectory))
        return tf.vectorized_map(lambda x: tf.math.reduce_sum(tf.cast((trajectory == x), tf.int32))
                                 > 0, tf_range)

    return tf.vectorized_map(_create_mask, trajectory)


def masked_argmax(tensor, mask):
    min = tf.math.reduce_min(tensor)
    return tf.argmax(tf.where(mask, min, tensor), axis=1, output_type=tf.int32)


def clipped_log(tensor):
    tensor = tf.clip_by_value(tensor, 1e-12, 1.0)
    return tf.math.log(tensor)


def int_and(x: tf.Tensor, y: tf.Tensor):
    return x * y


def int_or(x: tf.Tensor, y: tf.Tensor):
    return tf.cast(tf.cast(x + y, tf.bool), tf.int32)


def int_not(x: tf.Tensor):
    return (-1) * x + 1


def int_xor(x: tf.Tensor, y: tf.Tensor):
    return tf.math.mod(x + y, 2)


def sample_action(p: tf.Tensor):
    # B, N
    tf.assert_rank(p, 2)
    # B, 1
    return tf.squeeze(tf.random.categorical(tf.math.log(p), 1), axis=-1)
