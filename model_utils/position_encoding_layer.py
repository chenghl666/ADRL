from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow import keras


class PositionEncoding(keras.layers.Layer):
    def __init__(self, hidden_size, is_encoding, **kwargs):
        # Pass dtype=float32, as we have not yet tested if layer norm is numerically
        # stable in float16 and bfloat16.
        super(PositionEncoding, self).__init__(dtype="float32", **kwargs)
        self.hidden_size = hidden_size
        self.is_encoding = is_encoding

    def call(self, position, min_timescale=1.0, max_timescale=1.0e4):
        # Create binary mask of size [batch_size, n_visits, n_codes]
        if self.is_encoding:
            mask = tf.cast(tf.not_equal(position, 0), tf.float32)
        else:
            mask = tf.zeros(tf.shape(position))
        position = tf.cast(position, tf.float32)
        num_timescales = self.hidden_size // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale))
                                   /(tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, -1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
        signal *= tf.expand_dims(mask, -1)
        return signal
