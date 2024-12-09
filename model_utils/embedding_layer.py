from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class EmbeddingSharedWeights(keras.layers.Layer):
    """Calculates input embeddings"""

    def __init__(self, vocab_size, hidden_size, **kwargs):
        """Specify characteristic parameters of embedding layer.

        Args:
          vocab_size: Number of tokens in the embedding. (Typically ~32,000)
          hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, _):

          # Create and initialize weights. The random normal initializer was chosen
          # randomly, and works well.
          self.shared_weights = self.add_weight(
              name="weights", shape = (self.vocab_size, self.hidden_size),
              dtype='float32', trainable=True,
              initializer=tf.random_normal_initializer(0.0, self.hidden_size ** -0.5))
          # self.built = True

    def call(self, x):
        """Get token embeddings of x.

        Args:
          x: An int64 tensor with shape [batch_size, n_visits, n_codes]
        Returns:
          embeddings: float32 tensor with shape [batch_size, n_visits, n_codes, embedding_size]
          padding: float32 tensor with shape [batch_size, n_visits, n_codes] indicating the
            locations of the padding tokens in x.
        """
        # Create binary mask of size [batch_size, n_visits, n_codes]
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)
        #例如 embeddings=（32，10，128）=(batchsize,visits,embedding_size)
        embeddings = tf.gather(self.shared_weights, x)
        embeddings *= tf.expand_dims(mask, -1)



        # Scale embedding by the sqrt of the hidden size 按隐藏大小的sqrt平方根缩放嵌入 为什么？
        embeddings *= self.hidden_size ** 0.5

        return embeddings
