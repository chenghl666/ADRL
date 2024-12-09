"""Implementation of masked self-attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

from model_utils.attentionLayers import exp_mask_for_high_rank

import tensorflow as tf
from tensorflow import keras
from model_utils.attentionLayers import Flatten, Reconstruct


class Reshape(keras.layers.Layer):
    def __init__(self):
        super(Reshape, self).__init__()

    def call(self, inputs):
        v, ref, embedding_size = inputs
        batch_size = tf.shape(ref)[0]
        n_visits = tf.shape(ref)[1]
        out = tf.reshape(v, [batch_size, n_visits, embedding_size])
        return out


class DenseActivation(keras.layers.Layer):
    def __init__(self, output_size, activation=None):
        super(DenseActivation, self).__init__()
        self.output_size = output_size
        self.flatten = Flatten(1)
        self.linear = keras.layers.Dense(output_size, activation=activation)
        self.reconstruct = Reconstruct(1)


    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.linear(x)
        x = self.reconstruct([x,inputs])
        return x
class AttentionPooling(keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.dense = DenseActivation(embedding_size, activation='relu')
        self.linear = DenseActivation(embedding_size)
    def call(self, inputs):
        tensor, mask = inputs
        x = self.dense(tensor)
        x = self.linear(x)
        x = exp_mask_for_high_rank(x, mask)

        soft = tf.nn.softmax(x, 1)  # bs,skip_window,vec
        attn_output = tf.reduce_sum(soft * tensor, 1)  # bs, vec

        return attn_output