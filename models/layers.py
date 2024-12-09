import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, GRUCell
from tensorflow.keras.layers import Concatenate, Activation, RNN, StackedRNNCells
from tensorflow.keras.initializers import GlorotUniform

import config
from model_utils.normalization_layer import LayerNormalization


class MedicalEmbedding(Layer):
    def __init__(self, code_num, code_dim, name='code_embedding'):
        super().__init__(name=name)
        self.code_embeddings = self.add_weight(name='code_emb',
                                                  shape=(code_num, code_dim),
                                                  initializer=GlorotUniform(),
                                                  trainable=True)

    def call(self, inputs=None):
        return self.code_embeddings



class PatientEmbedding(Layer):
    def __init__(self, patient_num, patient_dim, name='patient_embedding'):
        super().__init__(name=name)
        self.patient_embeddings = self.add_weight(name='p_emb',
                                                  shape=(patient_num, patient_dim),
                                                  initializer=GlorotUniform(),
                                                  trainable=True)

    def call(self, inputs=None):
        return self.patient_embeddings


class GraphConvBlock(Layer):
    def __init__(self, node_type, dim, adj, name='graph_conv_block'):
        super().__init__(name=name)
        self.node_type = node_type
        self.adj = adj
        self.dense = Dense(dim, activation=None, name=name + '_dense')
        self.activation = Activation('relu', name=name + '_activation')
        self.bn = BatchNormalization(name=name + 'bn')

    def call(self, embedding, embedding_neighbor, weight_decay = None,real_US= None,real_V=None):
        output = embedding + tf.matmul(self.adj, embedding_neighbor)
        if self.node_type == 'code':
            assert weight_decay is not None
            US = tf.matmul(weight_decay, real_US)
            V = tf.matmul(weight_decay, real_V)
            output += 0.05*US @ (tf.transpose(V) @ embedding)  #0.05  0.1
        output = self.dense(output)
        output = self.bn(output)
        output = self.activation(output)
        return output


def norm_no_nan(x):
    return tf.math.divide_no_nan(x, tf.reduce_sum(x, axis=-1, keepdims=True))



class CommonConv(Layer):
    def __init__(self, layers, code_code_adj, name='common_conv'):
        super().__init__(name=name)
        self.layers = layers
        self.dense = Dense(128, activation=None, name=name + '_dense')
        self.activation = Activation('relu', name=name + '_activation')
        self.bn = LayerNormalization(name=name + 'bn')
        code_dim = 128



        self.real_US = self.add_weight(name='real_us',
                                       shape=(code_code_adj.shape[0], code_dim),
                                       initializer=GlorotUniform(),
                                       trainable=True)
        self.real_V = self.add_weight(name='real_v',
                                      shape=(code_code_adj.shape[0], code_dim),
                                      initializer=GlorotUniform(),#tf.keras.initializers.constant(real_V),
                                      trainable=True)


    def get_svd(self,norm_adj,hidden_size):

        # Perform SVD
        s, u, v = tf.linalg.svd(norm_adj)
        # Truncate to desired rank (assuming self.args.hidden_size is the desired rank)
        rank = hidden_size
        truncated_s = tf.linalg.diag(s[:rank])
        truncated_u = u[:, :rank]
        truncated_v = v[:, :rank]

        real_US = tf.matmul(truncated_u, truncated_s)
        real_V = truncated_v
        return real_US,real_V


    def call(self, adjacency, embedding,noise=True):


        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        if noise:
            US = tf.matmul(adjacency, self.real_US)
            V = tf.matmul(adjacency, self.real_V)

        for i in range(self.layers):
            if noise:
                emb = tf.matmul(adjacency, item_embeddings)
                output = emb + 0.05*US @ (tf.transpose(V) @ item_embeddings)
            else:
                output =  tf.matmul(adjacency, item_embeddings)

            final.append(output)

        result_item_embeddings = tf.reduce_sum(final, 0) / (self.layers + 1)
        result_item_embeddings = self.bn(result_item_embeddings)
        return result_item_embeddings


#Ablation experiment SVD
class CommonConv2(Layer):
    def __init__(self, layers, code_code_adj, name='common_conv'):
        super().__init__(name=name)
        self.layers = layers
        self.dense = Dense(128, activation=None, name=name + '_dense')
        self.activation = Activation('relu', name=name + '_activation')
        self.bn = LayerNormalization(name=name + 'bn')
        code_dim = 128

        self.real_US, self.real_V = self.get_svd(code_code_adj, code_dim)

    def get_svd(self,norm_adj,hidden_size):

        # Perform SVD
        s, u, v = tf.linalg.svd(norm_adj)
        # Truncate to desired rank (assuming self.args.hidden_size is the desired rank)
        rank = hidden_size
        truncated_s = tf.linalg.diag(s[:rank])
        truncated_u = u[:, :rank]
        truncated_v = v[:, :rank]

        real_US = tf.matmul(truncated_u, truncated_s)
        real_V = truncated_v
        return real_US,real_V

    def call(self, adjacency, embedding,noise=True):


        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            if noise:
                emb = tf.matmul(adjacency, item_embeddings)
                output = emb + 0.1*self.real_US @ (tf.transpose(self.real_V) @ item_embeddings)#0.05
            else:
                output =  tf.matmul(adjacency, item_embeddings)

            final.append(output)

        result_item_embeddings = tf.reduce_sum(final, 0) / (self.layers + 1)
        result_item_embeddings = self.bn(result_item_embeddings)
        return result_item_embeddings


class VisitEmbedding(Layer):
    def __init__(self, max_seq_len, name='visit_embedding'):
        super().__init__(name=name)
        self.max_seq_len = max_seq_len

    def call(self, code_embeddings, visit_codes, visit_lens):
        """
            visit_codes: (batch_size, max_seq_len, max_code_num_in_a_visit)
        """
        visit_codes_embedding = tf.nn.embedding_lookup(code_embeddings, visit_codes)  # (batch_size, max_seq_len, max_code_num_in_a_visit, code_dim)
        visit_codes_mask = tf.expand_dims(visit_codes > 0, axis=-1)
        visit_codes_mask = tf.cast(visit_codes_mask, visit_codes_embedding.dtype)
        visit_codes_embedding *= visit_codes_mask  # (batch_size, max_seq_len, max_code_num_in_a_visit, code_dim)
        visit_codes_num = tf.expand_dims(tf.reduce_sum(tf.cast(visit_codes > 0, visit_codes_embedding.dtype), axis=-1), axis=-1)
        visits_embeddings = tf.math.divide_no_nan(tf.reduce_sum(visit_codes_embedding, axis=-2), visit_codes_num)  # (batch_size, max_seq_len, code_dim)
        visit_mask = tf.expand_dims(tf.sequence_mask(visit_lens, self.max_seq_len, dtype=visits_embeddings.dtype), axis=-1)  # (batch_size, max_seq_len, 1)
        visits_embeddings *= visit_mask  # (batch_size, max_seq_len, code_dim)
        return visits_embeddings


def masked_softmax(inputs, mask):
    inputs = inputs - tf.reduce_max(inputs, keepdims=True, axis=-1)
    exp = tf.exp(inputs) * mask
    result = tf.math.divide_no_nan(exp, tf.reduce_sum(exp, keepdims=True, axis=-1))
    return result


class Attention(Layer):
    def __init__(self, attention_dim, name='attention'):
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.u_omega = self.add_weight(name=name + '_u', shape=(attention_dim,), initializer=GlorotUniform())
        self.w_omega = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.w_omega = self.add_weight(name=self.name + '_w', shape=(hidden_size, self.attention_dim), initializer=GlorotUniform())

    def call(self, x, mask=None):
        """
            x: (batch_size, max_seq_len, rnn_dim[-1] / hidden_size)
        """
        t = tf.matmul(x, self.w_omega)
        vu = tf.tensordot(t, self.u_omega, axes=1)  # (batch_size, max_seq_len)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = tf.nn.softmax(vu)  # (batch_size, max_seq_len)
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), axis=-2)  # (batch_size, rnn_dim[-1] / hidden_size)
        return output, alphas


class TemporalEmbedding(Layer):
    def __init__(self, rnn_dims, attention_dim, max_seq_len, cell_type=GRUCell, name='code_ra'):
        super().__init__(name=name)
        rnn_cells = [cell_type(rnn_dim) for rnn_dim in rnn_dims]
        stacked_rnn = StackedRNNCells(rnn_cells)
        self.rnn_layers = RNN(stacked_rnn, return_sequences=True, name=name + 'rnn')
        self.attention = Attention(attention_dim, name=name + '_attention')
        self.max_seq_len = max_seq_len

    def call(self, embeddings, lens):
        seq_mask = tf.sequence_mask(lens, self.max_seq_len, dtype=embeddings.dtype)
        outputs = self.rnn_layers(embeddings) * tf.expand_dims(seq_mask, axis=-1)  # (batch_size, max_seq_len, rnn_dim[-1])
        outputs, alphas = self.attention(outputs, seq_mask)  # (batch_size, rnn_dim[-1])
        return outputs, alphas


def log_no_nan(x):
    mask = tf.cast(x == 0, dtype=x.dtype)
    x = x + mask
    return tf.math.log(x)






