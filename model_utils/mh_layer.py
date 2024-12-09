from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras import regularizers

from model_utils.attentionLayers import Flatten


class TimeGSMKernel(tf.keras.layers.Layer):
    """Build a GSM kernel"""
    def __init__(self, time_dim=4, l1=0., l2=0.):
        """

        Args:
            time_dim: dimensionality of the hidden embedding of ffn
            l1: weight for l1 reg
            l2: weight for l2 reg
        """
        super(TimeGSMKernel, self).__init__()

        # ui
        self.period_var = gsm_ffn(time_dim, l1, l2)
        self.period_var.build(input_shape=(None, None, 1))
        # li
        self.sigma_var = gsm_ffn(time_dim, l1, l2)
        self.sigma_var.build(input_shape=(None, None, 1))
        # ai
        self.basis_expan_var = gsm_ffn(time_dim, l1, l2)
        self.basis_expan_var.build(input_shape=(None, None, 1))

        self.time_dim = time_dim

    def call(self, inputs):
        """

        Args:
            inputs: [t_q, t_k], i.e., inputs for building the GSM kernel
            scope: scope name

        Returns: the GSM kernel

        """
        inputs = tf.expand_dims(inputs, -1)
        inputs = tf.cast(inputs, dtype=tf.float32)
        q= inputs  # batch * Lq * 1
        k = inputs
        #print(q.get_shape().as_list())
        lq = q.get_shape().as_list()[2]
        lk = k.get_shape().as_list()[2]

        period_var_q = self.period_var(q)  # batch * lq * time_dim
        period_var_k = self.period_var(k)



        q_period_q = 2 * math.pi * tf.multiply(period_var_q, q)  # batch * lq * time_dim
        k_period_k = 2 * math.pi * tf.multiply(period_var_k, k)  # batch * lk * time_dim

        q_period_q = tf.tile(tf.expand_dims(q_period_q, axis=2), [1, 1, lk, 1])  # batch * lq * lk * time_dim
        k_period_k = tf.tile(tf.expand_dims(k_period_k, axis=1), [1, lq, 1, 1])  # batch * lq * lk * time_dim
        qk_period_diff = tf.keras.layers.Add()([q_period_q, -1. * k_period_k])  # batch * lq * lk * time_dim

        cos_enc = tf.cos(qk_period_diff)

        sigma_q = self.sigma_var(q)  # batch * lq * time_dim
        sigma_q = tf.tile(tf.expand_dims(sigma_q, axis=2), [1, 1, lk, 1])  # batch * lq * lk * time_dim
        sigma_q += 1e-6  # add an epsilon to avoid zeros
        sigma_k = self.sigma_var(k)  # batch * lk * time_dim
        sigma_k = tf.tile(tf.expand_dims(sigma_k, axis=1), [1, lq, 1, 1])  # batch * lq * lk * time_dim
        sigma_k += 1e-6  # add an epsilon to avoid zeros

        qk_diff = tf.keras.layers.Add()(
            [tf.tile(q, [1, 1, lk]), -1. * tf.transpose(tf.tile(k, [1, 1, lq]), perm=[0, 2, 1])]
        )  # batch * lq * lk
        qk_diff = tf.expand_dims(qk_diff, axis=-1)  # batch * lq * lk *1
        qk_diff = tf.tile(qk_diff, [1, 1, 1, self.time_dim])  # batch*lq*lk*time_dim

        exp_enc = tf.exp(-1.*tf.divide(qk_diff**2, tf.add(sigma_q**2, sigma_k**2)))
        local_enc = (tf.divide(2 * tf.multiply(sigma_q, sigma_k), tf.add(sigma_q**2, sigma_k**2)))**0.5

        gibbs_enc = tf.multiply(local_enc, exp_enc)
        basis_expan_q = self.basis_expan_var(q)
        basis_expan_k = self.basis_expan_var(k)

        basis_expan_q = tf.tile(tf.expand_dims(basis_expan_q, axis=2), [1, 1, lk, 1])
        basis_expan_k = tf.tile(tf.expand_dims(basis_expan_k, axis=1), [1, lq, 1, 1])

        basis_expan_qk = tf.multiply(basis_expan_q, basis_expan_k)

        time_gsm_kernel = tf.multiply(basis_expan_qk, tf.multiply(gibbs_enc, cos_enc))
        time_gsm_kernel = tf.reduce_sum(time_gsm_kernel, axis=-1)

        return time_gsm_kernel

def gsm_ffn(time_dim, l1=0., l2=0.):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(time_dim, activation='relu', kernel_initializer='glorot_uniform',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(time_dim, activation='relu', kernel_initializer='glorot_uniform',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    ])
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self,direction, train, dropout, num_units, num_heads=10,time_dim=1,  **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.direction = direction
        self.dropout = dropout
        self.train = train
        self.num_units = num_units
        self.time_dim = time_dim
        self.q_linear = keras.layers.Dense(self.num_units, use_bias=False)
        self.k_linear = keras.layers.Dense(self.num_units, use_bias=False)
        self.v_linear = keras.layers.Dense(self.num_units, use_bias=False)

        self.time_dim = time_dim
        self.time_kernel = None
        l1 = 0.
        l2 = 0.
        if self.time_dim is not None and self.time_dim != 1:
            self.time_kernel = TimeGSMKernel(self.time_dim, l1, l2)




    def call(self, inputs,training = True):

        # because of self-attention, queries and keys is equal to inputs
        query_input,input_tensor, input_mask, time_inputs,att_bias = inputs
        queries = query_input
        keys = input_tensor

        # Linear projections
        Q = self.q_linear(queries)  # (N, L_q, d)
        K = self.k_linear(keys)  # (N, L_k, d)
        V = self.v_linear(keys)  # (N, L_k, d)

        # print('Q shape: ', Q.get_shape())

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, L_q, d/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, L_k, d/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, L_k, d/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, L_q, L_k)

        # Scale  例如dk = 128/4=32
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) # (h*N, L_q, L_k)





        ######################################################################
        # time related
        time_kernel_vals = None

        if time_inputs is not None and self.time_kernel is not None:
            time_kernel_vals = self.time_kernel(time_inputs)

        if time_kernel_vals is not None:
            time_kernel_vals = tf.tile(tf.expand_dims(time_kernel_vals, axis=1), [1, self.num_heads, 1, 1])
            time_kernel_vals = time_kernel_vals / (K_.get_shape().as_list()[-1] ** 0.5)
            time_kernel_vals = Flatten(2, name='code_flaten22')(time_kernel_vals)
            outputs += time_kernel_vals
        ################################################################################################




        key_masks = tf.sign(tf.reduce_sum(tf.abs(K_), axis=-1))  # (h*N, T_k)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, T_k)
        key_masks = tf.tile(key_masks, [1, tf.shape(Q_)[1], 1])  # (h*N, T_q, T_k)

        # Apply masks to outputs #padding
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # exp mask
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        #n_visits = input_tensor.get_shape()[1]
        n_visits = tf.shape(input_tensor)[1]
        sw_indices = tf.range(n_visits, dtype=tf.int32)
        sw_col, sw_row = tf.meshgrid(sw_indices, sw_indices)
        if self.direction == 'diag':
            # shape of (n_visits, n_visits)
            attention_mask = tf.cast(tf.linalg.diag(- tf.ones([n_visits], tf.int32)) + 1, tf.bool)
        elif self.direction == 'forward':
            attention_mask = tf.greater(sw_row, sw_col)  # shape of (n_visits, n_visits)
        else:
            attention_mask = tf.greater(sw_col, sw_row)  # shape of (n_visits, n_visits)
        adder = (1.0 - tf.cast(attention_mask, outputs.dtype)) * -10000.0

        if self.direction != 'diag':
            outputs += adder


        # softmax
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(Q_), axis=-1))  # (h*N, T_q)
        query_masks = tf.expand_dims(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(K_)[1]])  # (h*N, T_q, T_k)

        # Apply masks to outputs
        outputs = outputs * query_masks

        # Dropouts
        if training:
            outputs = tf.nn.dropout(outputs, rate=self.dropout)
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, L_q, d)

        # input padding
        val_mask = tf.expand_dims(input_mask, -1)
        outputs = tf.multiply(outputs, tf.cast(val_mask, tf.float32), name='mask_for_high_rank')

        return outputs