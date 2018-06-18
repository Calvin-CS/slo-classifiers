"""Transformer keras implementation.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
    Jones, L., Gomez, A. N., … Polosukhin, I. (2017).
Attention Is All You Need.
In The 31st Conference on Neural Information Processing Systems (pp. 48–54).
Retrieved from http://arxiv.org/abs/1706.03762

The original code is Lsdefine/attention-is-all-you-need-keras on GitHub.
"""
import logging
from itertools import permutations

import numpy as np

import keras.backend as K
from keras.initializers import Ones, Zeros
from keras.layers import (Activation, Add, Concatenate, Conv1D, Dense, Dropout,
                          Embedding, Flatten, Input, Lambda, Layer,
                          TimeDistributed)
from keras.models import Model
from keras.optimizers import Nadam

logger = logging.getLogger(__name__)


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(
            x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    # NOTE: mode 0 does not work (cannot infer output shapes correctly)
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=1):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(
                    TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(
                    TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(
                    TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            qs = Lambda(self._reshape1)(qs)
            ks = Lambda(self._reshape1)(ks)
            vs = Lambda(self._reshape1)(vs)

            mask = Lambda(lambda x: K.repeat_elements(x, self.n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            head = Lambda(self._reshape2)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(self.n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads)
            attn = Concatenate()(attns)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn

    def _reshape1(self, x):
        s = K.shape(x)   # [batch_size, len_q, n_head * d_k]
        x = K.reshape(x, [s[0], s[1], self.n_head, self.d_k])
        x = K.permute_dimensions(x, [2, 0, 1, 3])
        # [n_head * batch_size, len_q, d_k]
        x = K.reshape(x, [-1, s[1], self.d_k])
        return x

    def _reshape2(self, x):
        s = K.shape(x)   # [n_head * batch_size, len_v, d_v]
        x = K.reshape(x, [self.n_head, -1, s[1], s[2]])
        x = K.permute_dimensions(x, [1, 2, 0, 3])
        # [batch_size, len_v, n_head * d_v]
        x = K.reshape(x, [-1, s[1], self.n_head * self.d_v])
        return x


class PositionwiseFeedForward():
    def __init__(self, d_hid, dim_pff, dropout=0.1):
        self.w_1 = Conv1D(dim_pff, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, dim_pff, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(
            d_model, dim_pff, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(
            enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class ExtraEncoderLayer():
    """Using given encoder input as key-value pairs, accept extra information as query."""

    def __init__(self, d_model, dim_pff, n_head, d_k, d_v, dropout=0.1, self_att=True):
        self.self_att = self_att
        if self.self_att:
            self.self_att_layer = MultiHeadAttention(
                n_head, d_model, d_k, d_v, dropout=dropout)
        self.xtra_att_layer = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(
            d_model, dim_pff, dropout=dropout)

    def __call__(self, query, enc_output, self_mask=None, enc_mask=None):
        if self.self_att:
            output, slf_attn = self.self_att_layer(
                query, query, query, mask=self_mask)
        output, xtra_attn = self.xtra_att_layer(
            output, enc_output, enc_output, mask=enc_mask)
        output = self.pos_ffn_layer(output)
        return (output, slf_attn, xtra_attn) if self.self_att else (output, xtra_attn)


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def get_pad_mask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)  # q_len * q_len, all 1
    # k_len * k_len, 1.0 (not padding) or 0.0 (padding)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


class Encoder():
    def __init__(self, d_model, dim_pff, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.d_model = d_model
        self.layers = [EncoderLayer(
            d_model, dim_pff, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        x = self.emb_layer(src_seq)
        x = Lambda(lambda x: x * np.sqrt(self.d_model))(x)
        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            x = Add()([x, pos])
        if return_att:
            atts = []
        mask = Lambda(lambda x: get_pad_mask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att:
                atts.append(att)
        return (x, atts) if return_att else x


class ExtraEncoder():
    def __init__(self, d_model, dim_pff, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, self_att=True):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.d_model = d_model
        self.self_att = self_att
        self.layers = [ExtraEncoderLayer(
            d_model, dim_pff, n_head,
            d_k, d_v, dropout, self_att=self.self_att) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output,
                 return_att=False, active_layers=999):
        q = self.emb_layer(tgt_seq)
        q = Lambda(lambda x: x * np.sqrt(self.d_model))(q)
        if tgt_pos is not None:
            pos_q = self.pos_layer(tgt_pos)
            q = Add()([q, pos_q])
        if return_att:
            self_atts, xtra_atts = [], []
        mask_q = Lambda(lambda x: get_pad_mask(x, x))(tgt_seq)
        mask_enc = Lambda(lambda x: get_pad_mask(
            x[0], x[1]))([tgt_seq, src_seq])
        for xenc_layer in self.layers[:active_layers]:
            if self.self_att:
                output, self_att, xtra_att = xenc_layer(
                    q, enc_output, self_mask=mask_q, enc_mask=mask_enc)
            else:
                output, xtra_att = xenc_layer(
                    q, enc_output, self_mask=mask_q, enc_mask=mask_enc)
            if return_att:
                self_atts.append(self_att)
                xtra_atts.append(xtra_att)
        return (output, self_atts, xtra_att) if return_att else output


def get_pos_seq(x):
    mask = K.cast(K.not_equal(x, 0), 'int32')
    pos = K.cumsum(K.ones_like(x, 'int32'), 1)
    return pos * mask


# def get_loss(args):
#     y_pred, y_true = args
#     y_true = tf.cast(y_true, 'int32')
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#     mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
#     loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
#     loss = K.mean(loss)
#     return loss


# def get_accu(args):
#     y_pred, y_true = args
#     mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
#     corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
#     corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
#     return K.mean(corr)


def build_model(embedding_matrix=None,
                word_index=None, max_vocabsize=200_000,
                max_seqlen=20, max_tgtlen=4, dim_wordvec=64,
                profile=False, max_prflen=20,
                m_profile=1, target=False, parallel=False,
                xtra_self_att=True,
                dropout=0.1, dim_output=3, lr=0.001,
                dim_pff=None, num_head=4, num_layers=1, **kwargs):
    """The builder function to compile Transformer.

    Args (Common in neural models):
        embedding_matrix: word embedding matrix of pretrained word vectors
        word_index: a dictionary of word-to-ID mapping
        max_vocabsize: the maximum number of words to consider
        max_seqlen: the maximum length in words of each input text (body)
        max_tgtlen: the maximum length in words of each input text (target)
        dim_wordvec: the dimensionality of word vectors. If embedding_matrix specified, this value is ignored and replaced with the dimension of embedding_matrix.
        profile: whether to incorporate profile feature.
        max_prflen: the maximum length in words of each input text (profile)
        dim_output: the number of class labels to predict
        lr: the initial learning rate for NAdam
        dropout: the ratio of dropout through out in this model

    Args (Transformer Specific):
        m_profile: the direction of attention between tweet text and profile.
            if profile is True, input 0 is fed to XEnc and
            input 1 is used for the attention source.
            The order of inputs is two-fold.

            i - [input0, input1]
            -------------------
            1 - [profile, tweet] (default)
            2 - [tweet, profile]
        target: whether to incorporate target feature.
            this option is valid only if profile is False.
            if False, the target input is ignored.
            because the order of inputs can be two-fold,
            two options are available.
            input 0 is fed to XEnc and
            input 1 is used for the attention source.

            i - [input0, input1]
            -------------------
            1 - [target,  tweet]
            2 - [tweet,  target]
        parallel: the multi-column way to incorporate target and profile,
            if both of profile and target are True
            input 0 is fed to XEnc as query, input 1 and input 2 are used as attention sources.
            specify serial modes by the number [1--6]

            i - [input0, input1, input2]
            -------------------------------------
            1 - [target, tweet, profile] (default)
            2 - [target, profile, tweet]
            3 - [tweet, target, profile]
            4 - [tweet, profile, target]
            5 - [profile, target, tweet]
            6 - [profile, tweet, target]
        xtra_self_att: if True, XEnc is built with self attention sub-layer
        dim_pff: the dimensionality of Pointwise Feed Forward layers
        num_head: the number of heads in Multi Head Attention leyers
        num_layers: the number of stacks of the Transformer (Encoder) layer

    Returns:
        compiled keras.models.Model
    """
    # TODO: profile incorporation
    logger.debug('Compiling Transformer...')

    # params
    vocab_size = min(max_vocabsize, len(word_index)) + 1
    if embedding_matrix is not None:
        dim_wordvec = embedding_matrix.shape[1]
    d_k = dim_wordvec // num_head
    d_v = dim_wordvec // num_head
    if dim_pff is None:
        dim_pff = dim_wordvec * 4  # the setting of original paper

    src_seq_input = Input(shape=(max_seqlen,), dtype='int32', name='s_input')
    # target input is always included
    tgt_seq_input = Input(shape=(max_tgtlen,), dtype='int32', name='t_input')
    src_pos = Lambda(get_pos_seq, name='PosSeq')(src_seq_input)
    if target:
        tgt_pos = Lambda(get_pos_seq, name='PosSeqT')(tgt_seq_input)
    if profile:
        prf_seq_input = Input(shape=(max_prflen,),
                              dtype='int32', name='p_input')
        prf_pos = Lambda(get_pos_seq, name='PosSeqP')(prf_seq_input)

    if embedding_matrix is None:
        s_word_emb = Embedding(output_dim=dim_wordvec,
                               input_dim=vocab_size,
                               # input_length=max_seqlen,  # don't specify this, otherwise confuse keras
                               trainable=True, name='InputEnc')
        # if below enabled, separate embedding weights in text and target
        # t_word_emb = Embedding(output_dim=dim_wordvec,
        #                        input_dim=vocab_size,
        #                        # input_length=max_tgtlen,
        #                        trainable=True, name='InputEncT')
    else:
        s_word_emb = Embedding(output_dim=dim_wordvec,
                               input_dim=vocab_size,
                               weights=[embedding_matrix],
                               trainable=False, name='InputEnc')

    pos_emb = Embedding(output_dim=dim_wordvec,
                        # input_dim=vocab_size,
                        input_dim=max_seqlen + 1,  # for index 0?
                        # input_length=max_seqlen,  # don't specify this, otherwise confuse keras
                        weights=[get_pos_encoding_matrix(
                            max_seqlen + 1, dim_wordvec)],
                        trainable=False, name='PosEnc')

    encoder = Encoder(dim_wordvec, dim_pff, num_head, d_k, d_v,
                      num_layers, dropout,
                      word_emb=s_word_emb, pos_emb=pos_emb)
    if bool(target) ^ bool(profile):  # XOR: one XEnc
        xencoder = ExtraEncoder(dim_wordvec, dim_pff, num_head,
                                d_k, d_v,
                                num_layers, dropout,
                                word_emb=s_word_emb, pos_emb=pos_emb,
                                self_att=xtra_self_att)
        if target:
            if target == 1:  # target as query
                logger.debug('target incorporation mode 1')
                enc_output = encoder(src_seq_input, src_pos)
                enc_output = xencoder(tgt_seq_input, tgt_pos,
                                      src_seq_input, enc_output)
            elif target == 2:  # tweet as query
                logger.debug('target incorporation mode 2')
                enc_output = encoder(tgt_seq_input, tgt_pos)
                enc_output = xencoder(src_seq_input, src_pos,
                                      tgt_seq_input, enc_output)
            else:
                raise ValueError(
                    "specify the target incorporation mode from 1 or 2")
        if profile:
            if m_profile == 1:  # profile as query
                logger.debug('profile incorporation mode 1')
                enc_output = encoder(src_seq_input, src_pos)
                enc_output = xencoder(prf_seq_input, prf_pos,
                                      src_seq_input, enc_output)
            elif m_profile == 2:  # tweet as query
                logger.debug('profile incorporation mode 2')
                enc_output = encoder(prf_seq_input, prf_pos)
                enc_output = xencoder(src_seq_input, src_pos,
                                      prf_seq_input, enc_output)
            else:
                raise ValueError(
                    "specify the profile incorporation mode from 1 or 2")
    elif target and profile:  # 2 XEnc and 2 Enc
        encoder2 = Encoder(dim_wordvec, dim_pff, num_head, d_k, d_v,
                           num_layers, dropout,
                           word_emb=s_word_emb, pos_emb=pos_emb)
        xencoder1 = ExtraEncoder(dim_wordvec, dim_pff, num_head,
                                 d_k, d_v,
                                 num_layers, dropout,
                                 word_emb=s_word_emb, pos_emb=pos_emb,
                                 self_att=xtra_self_att)
        xencoder2 = ExtraEncoder(dim_wordvec, dim_pff, num_head,
                                 d_k, d_v,
                                 num_layers, dropout,
                                 word_emb=s_word_emb, pos_emb=pos_emb,
                                 self_att=xtra_self_att)
        combs = list(permutations(
            (tgt_seq_input, src_seq_input, prf_seq_input)))
        poss = list(permutations((tgt_pos, src_pos, prf_pos)))
        input0, input1, input2 = combs[parallel - 1]
        pos0, pos1, pos2 = poss[parallel - 1]
        logger.debug(f'parallel model mode {parallel}')
        # first unit
        enc_output1 = encoder(input1, pos1)
        enc_output1 = xencoder1(
            input0, pos0, input1, enc_output1)
        # second unit
        enc_output2 = encoder2(input2, pos2)
        enc_output2 = xencoder2(
            input0, pos0, input2, enc_output2)
        # Merge two units
        enc_output = Concatenate()([enc_output1, enc_output2])
    else:  # t == False and p == False: tweet only
        enc_output = encoder(src_seq_input, src_pos)

    # Prediction Layer
    # 1. transform len_q * d_model matrix into len_q * num_head vector
    # 2. feed the vector into perceptron (1-leyer FeedForward)
    # NOTE: dropout, multilayer perceptron can be applied further
    output = TimeDistributed(Dense(num_head, activation='softmax'))(enc_output)
    output = Dropout(dropout)(output)
    output = Flatten()(output)
    output = Dense(dim_output, activation='softmax')(output)

    if profile:
        model = Model(inputs=(src_seq_input, tgt_seq_input,
                              prf_seq_input), outputs=output)
    else:
        model = Model(inputs=(src_seq_input, tgt_seq_input), outputs=output)

    nadam = Nadam(lr=lr)
    model.compile(
        loss='categorical_crossentropy',  # need one-hot-encoding on y
        # loss='sparse_categorical_crossentropy',  # accept 1-dim y; slow?
        optimizer=nadam,
        metrics=['accuracy']
    )
    logger.debug('...Compile done.')
    # model.summary()

    return model
