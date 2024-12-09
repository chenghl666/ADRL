import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from model_utils import embedding_layer
from utilss.configs import cfg


class CommonModel(object):

    def __init__(self, dataset):

        # ------ start ------
        self.lr = 0.0001
        self.dropout_rate = cfg.dropout
        self.n_intervals = dataset.days_size
        self.n_visits = cfg.valid_visits
        self.n_codes = dataset.max_len_visit
        self.vocabulary_size = len(dataset.dictionary)
        self.digit3_size = len(dataset.dictionary_3digit)
        self.pos_encoding = cfg.pos_encoding
        self.embedding_size = cfg.embedding_size
        self.num_hidden_layers = cfg.num_hidden_layers
        self.train = cfg.train
        self.predict_type = cfg.predict_type
        self.model = None
        self.direction = cfg.direction

        # ---- place holder -----
        # tensor with shape [batch_size, n_visits, n_codes]
        self.code_inputs = keras.layers.Input(shape=(self.n_visits, self.n_codes,), dtype=tf.int32, name='train_inputs')
        self.interval_inputs = keras.layers.Input(shape=(self.n_visits,), dtype=tf.int32, name='interval_inputs')

        # tensor with shape [batch_size, n_visits, n_codes]
        self.inputs_mask = math_ops.not_equal(self.code_inputs, 0)
        # tensor with shape [batch_size, n_visits]
        visit_mask = tf.reduce_sum(tf.cast(self.code_inputs, tf.int32), -1)
        self.visit_mask = tf.cast(visit_mask, tf.bool)

        self.embedding = embedding_layer.EmbeddingSharedWeights(self.vocabulary_size,
                                                                self.embedding_size,name='codes_embedding')
        self.params = dict()
        self.params["hidden_size"] = self.embedding_size
        self.params["dropout"] = self.dropout_rate
        self.params["direction"] = 'diag'
        self.params["num_heads"] = cfg.num_heads
        self.params["num_hidden_layers"] = cfg.num_hidden_layers
        self.params["allow_ffn_pad"] = False

    def build_network(self):
        pass

