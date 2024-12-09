import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Concatenate
from tensorflow.python.ops import math_ops

from model_utils import common_layer
from model_utils.attentionLayers import Flatten
from model_utils.normalization_layer import LayerNormalization
from models.layers import PatientEmbedding, MedicalEmbedding, CommonConv, \
    norm_no_nan, Attention

from models.layers import VisitEmbedding, TemporalEmbedding


class ADRLFeatureExtractor(Layer):
    def __init__(self, config, hyper_params, name='adrl_feature'):
        super().__init__(name=name)
        self.config = config
        self.hyper_params = hyper_params

        self.params = dict()
        self.params['allow_ffn_pad'] = False
        self.params['direction'] = 'forward'
        self.params['dropout'] = 0.4
        self.params['filter_size'] = 128
        self.params['hidden_size'] = 128
        self.params['is_scale'] = False
        self.params['num_heads'] = 1
        self.params['num_hidden_layers'] = 1

        self.layer_norm1 = LayerNormalization()

        self.embedding_layer = MedicalEmbedding(config['code_num'], code_dim=128)


        self.code_code_adj = config['code_code_adj']
        self.common_conv = CommonConv(3, self.code_code_adj)


        self.visit_embedding_layer = VisitEmbedding(
            max_seq_len=config['max_visit_seq_len'])
        self.visit_temporal_embedding_layer = TemporalEmbedding(
            rnn_dims=hyper_params['visit_rnn_dims'],
            attention_dim=hyper_params['visit_attention_dim'],
            max_seq_len=config['max_visit_seq_len'],
            name='visit_temporal')

        self.params['direction'] = 'diag'
        self.disease_encoder = common_layer.EncoderStack(self.params, True, "notlatest", 128, name='Vanilla_encoder')

        self.attention_code = Attention(64, name=name + '_code_attention')

        self.dropout = tf.keras.layers.Dropout(0.2)
        #self.dense_visit = Dense(200, use_bias=False)
        self.dense_visit = Dense(200,use_bias=False)

    def symmetric_normalize(matrix):
        # Calculate the inverse square root of the diagonal elements.
        diag_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(tf.linalg.diag_part(matrix)))

        # Perform symmetric normalization on the matrix.
        normalized_matrix = tf.matmul(tf.matmul(diag_inv_sqrt, matrix), diag_inv_sqrt)

        return normalized_matrix

    def call(self, inputs, training=True):
        visit_codes = inputs['visit_codes']  # (batch_size, max_seq_len, max_code_num_in_a_visit)
        visit_lens = tf.reshape(inputs['visit_lens'], (-1, ))  # (batch_size, )
        code_embeddings = self.embedding_layer(None)


        global_code_embs1 = self.common_conv(self.code_code_adj , code_embeddings,noise=False)
        global_code_embs2 = self.common_conv(self.code_code_adj, code_embeddings, noise=True)



        if training:
            # Incorporate structural contrastive learning.
            disease_ids  =visit_codes[visit_codes > 0]
            ssl_loss = self.ssl_layer_loss(global_code_embs2, global_code_embs1, disease_ids)
            self.add_loss(0.005*ssl_loss)#0.005 0.01


        inputs_mask = math_ops.not_equal(visit_codes, 0)
        e_mask = Flatten(1, name='mask_flaten')(inputs_mask)
        visit_codes_embedding = tf.nn.embedding_lookup(code_embeddings, visit_codes)
        visit_codes_embedding2 = tf.nn.embedding_lookup(global_code_embs1, visit_codes)
        visit_codes_mask = tf.expand_dims(visit_codes > 0, axis=-1)
        visit_codes_mask = tf.cast(visit_codes_mask, visit_codes_embedding.dtype)
        visit_codes_embedding *= visit_codes_mask
        visit_codes_embedding2 *= visit_codes_mask


        e = Flatten(2, name='code_flaten')(visit_codes_embedding)
        h = self.disease_encoder((e, e, e_mask, None,None))
        h = tf.reshape(h, [tf.shape(visit_codes)[0], tf.shape(visit_codes)[1], tf.shape(h)[1], tf.shape(h)[2]])
        h = tf.add(h, visit_codes_embedding)
        h = self.layer_norm1(h)
        h_ = tf.concat([h,visit_codes_embedding2],axis=3)
        h_ = tf.reshape(h_, [-1, h_.shape[2], h_.shape[3]])
        e_mask = tf.cast(e_mask, h.dtype)


        #First, concatenate local and global information, then use the attention mechanism to obtain visit embedding vectors
        visit = self.attention_code(h_, e_mask)[0]
        visit = tf.reshape(visit, [tf.shape(inputs_mask)[0], tf.shape(inputs_mask)[1],-1])



        visit_output, alpha_visit = self.visit_temporal_embedding_layer(visit, visit_lens)

        #Extract the last visit
        patient_num = visit_lens.shape[0]
        indices = [(i, int(visit_lens[i] - 1)) for i in range(patient_num)]
        last_visit = tf.gather_nd(visit, indices)

        visit_output = tf.concat([visit_output,last_visit],axis=1)
        visit_output = self.dense_visit(visit_output)

        output = visit_output
        return output

    def ssl_layer_loss(self, current_embedding, previous_embedding, disease_ids):
        ssl_temp = 0.1
        current_user_embeddings = tf.gather(current_embedding, disease_ids)
        previous_user_embeddings = tf.gather(previous_embedding, disease_ids)
        norm_user_emb1 = tf.math.l2_normalize(current_user_embeddings,
                                              axis=-1)
        norm_user_emb2 = tf.math.l2_normalize(previous_user_embeddings, axis=-1)
        norm_all_user_emb = tf.math.l2_normalize(previous_embedding, axis=-1)
        pos_score_user = tf.multiply(norm_user_emb1, norm_user_emb2)
        pos_score_user = tf.reduce_sum(pos_score_user, axis=1)
        ttl_score_user = tf.matmul(norm_user_emb1, tf.transpose(norm_all_user_emb))
        pos_score_user = tf.exp(pos_score_user / ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / ssl_temp), axis=1)
        ssl_loss_user = - tf.reduce_sum(tf.math.log(pos_score_user / ttl_score_user), axis=-1)
        return ssl_loss_user


class Classifier(Layer):
    def __init__(self, output_dim, activation=None, name='classifier'):
        super().__init__(name=name)
        self.dense = Dense(output_dim, activation=activation)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        return output


class ADRL(Model):
    def __init__(self, config, hyper_params, name='adrl'):
        super().__init__(name=name)
        self.adrl_feature_extractor = ADRLFeatureExtractor(config, hyper_params)
        self.classifier = Classifier(config['output_dim'], activation=config['activation'])

    def call(self, inputs, training=True):
        output = self.adrl_feature_extractor(inputs, training=training)
        output = self.classifier(output)
        output = tf.math.sigmoid(output)
        return output
