import tensorflow as tf
import numpy as np


def huber_loss(y_true, y_pred, weight=1.0, k=1.0):
    """Define a huber loss  https://en.wikipedia.org/wiki/Huber_loss
      tensor: tensor to regularize.
      k: value of k in the huber loss
      scope: Optional scope for op_scope.

    Huber loss:
    f(x) = if |x| <= k:
              0.5 * x^2
           else:
              k * |x| - 0.5 * k^2

    Returns:
      the L1 loss op."""
    print("y_pred,y_true", y_pred,y_true)
    d = tf.math.subtract(y_pred,y_true)
    a = tf.abs(d)
    loss = tf.where(a<k, 0.5*tf.square(d), k*a - 0.5 * k**2)
    # return tf.losses.compute_weighted_loss(losses,weight)
    return loss



