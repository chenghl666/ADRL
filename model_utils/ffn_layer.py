from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class FeedForwardNetwork(keras.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad, **kwargs):
    super(FeedForwardNetwork, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train
    self.allow_pad = allow_pad

    self.filter_dense_layer = keras.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = keras.layers.Dense(
        hidden_size, use_bias=True, name="output_layer")

  def call(self, x, training = True):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    query_input, x, mask, time_inputs,att_bias = x
    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, self.relu_dropout)
    output = self.output_dense_layer(output)

    # input padding
    # val_mask = tf.expand_dims(mask, -1)
    # output = tf.multiply(output, tf.cast(val_mask, tf.float32))

    return output
