import tensorflow as tf
from functools import reduce


def tf_kron_prod(a, b):
    res = tf.compat.v1.einsum('ij,ik->ijk', a, b)
    res = tf.compat.v1.reshape(res, [-1, tf.compat.v1.reduce_prod(res.shape[1:])])
    return res


def tf_bin(x, cut_points, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.get_shape().as_list()[0]
    W = tf.compat.v1.reshape(tf.compat.v1.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points = tf.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = tf.compat.v1.cumsum(tf.compat.v1.concat([tf.compat.v1.constant(0.0, shape=[1]), -cut_points], 0))
    h = tf.compat.v1.matmul(x, W) + b
    res = tf.compat.v1.nn.softmax(h / temperature)
    return res


def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(tf_kron_prod,
                  map(lambda z: tf_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
    return tf.compat.v1.matmul(leaf, leaf_score)
