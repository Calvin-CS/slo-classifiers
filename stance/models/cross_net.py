"""CrossNet layers and model build function."""
import logging

import keras.backend as K
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Embedding, Input, concatenate
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential

# from keras.wrappers.scikit_learn import KerasClassifier

# from sklearn.pipeline import Pipeline  # FeatureUnion,

logger = logging.getLogger(__name__)


class PredictLayer(object):
    def __init__(self, dense_dim, input_dim=0, dim_output=3, dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(dense_dim,
                             activation='relu',
                             input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(dim_output, activation='softmax'))

    def __call__(self, inputs):
        return self.model(inputs)


class AspectAttentionLayer(Layer):
    def __init__(self, n_reason=5, hidden_d=100, **kwargs):
        self.n_reason = n_reason
        self.hidden_d = hidden_d
        super(AspectAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        embedding_size = input_shape[-1]
        self.W1 = self.add_weight(shape=(embedding_size, self.hidden_d),
                                  name='W1',
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(shape=(self.hidden_d, self.n_reason),
                                  name='W2',
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        H = inputs[0]
        A1 = K.tanh(K.dot(H, self.W1))
        A = K.softmax(K.dot(A1, self.W2))
        M = K.batch_dot(K.permute_dimensions(A, (0, 2, 1)), H)
        m_merge = K.max(M, axis=1)
        return m_merge

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = {'n_reason': self.n_reason,
                  'hidden_d': self.hidden_d}
        base_config = super(AspectAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model(embedding_matrix=None,
                word_index=None, max_vocabsize=200_000,
                max_seqlen=20, max_tgtlen=4, dim_wordvec=64,
                profile=False, max_prflen=20,
                dropout=0.1, dim_output=3, lr=0.001,
                dim_lstm=100, num_reason=1, dim_dense=300, **kwargs):
    """The builder function to compile CrossNet.

    Args (Common in neural models):
        embedding_matrix: word embedding matrix of pretrained word vectors
        word_index: a dictionary of word-to-ID mapping
        max_vocabsize: the maximum number of words to consider
        max_seqlen: the maximum length in words of each input text (body)
        max_tgtlen: the maximum length in words of each input text (target)
        dim_wordvec: the dimensionality of word vectors. If embedding_matrix specified, this value is ignored and replaced with the dimension of embedding_matrix.
        profile: whether to incorporate profile feature
        max_prflen: the maximum length in words of each input text (profile)
        dim_output: the number of class labels to predict
        lr: the initial learning rate for NAdam
        dropout: the ratio of dropout through out in this model

    Args (CrossNet Specific):
        dim_lstm: the output dimensionality of LSTM for context/target encoding layers
        num_reason: the number of reasons in the aspect attention layer
        dim_dense: the hidden dimensionality of the aspect attention layer

    Returns:
        compiled keras.models.Model
    """
    logger.debug('Compiling CrossNet...')

    # Parameters
    vocab_size = min(max_vocabsize, len(word_index)) + 1
    if embedding_matrix is not None:
        dim_wordvec = embedding_matrix.shape[1]

    # Input layer
    s = Input(shape=(max_seqlen,), dtype='int32', name='s_input')
    t = Input(shape=(max_tgtlen,), dtype='int32', name='t_input')
    if profile:
        p = Input(shape=(max_prflen,), dtype='int32', name='p_input')

    # Embedding Layer
    if embedding_matrix is not None:
        emb = Embedding(output_dim=dim_wordvec,
                        input_dim=vocab_size,
                        weights=[embedding_matrix],
                        trainable=False)
    else:
        emb = Embedding(output_dim=dim_wordvec,
                        input_dim=vocab_size,
                        trainable=True)
    s_rep = emb(s)
    t_rep = emb(t)
    if profile:
        p_rep = emb(p)

    # keras v2.1.3+, Bidirectional can input/output internal states
    # c.f. https://stackoverflow.com/questions/47923370/
    target_context = Bidirectional(LSTM(dim_lstm,
                                        dropout=dropout,
                                        recurrent_dropout=dropout,
                                        return_state=True,
                                        return_sequences=False),
                                   merge_mode=None,
                                   input_shape=(max_tgtlen, K.int_shape(t_rep)[-1],))
    _, _, t_h_fw, t_c_fw, t_h_bw, t_c_bw = target_context(t_rep)
    context_encoding_layer = Bidirectional(LSTM(dim_lstm,
                                                unroll=True,
                                                dropout=dropout,
                                                recurrent_dropout=dropout,
                                                return_state=False,
                                                return_sequences=True),
                                           merge_mode='concat',
                                           input_shape=(max_seqlen, K.int_shape(s_rep)[-1],))
    sent_context = context_encoding_layer(
        s_rep, initial_state=[t_h_fw, t_c_fw, t_h_bw, t_c_bw])

    # Aspect Attention Layer
    aspect_attention_layer = AspectAttentionLayer(
        n_reason=num_reason, hidden_d=dim_dense)
    aspect_repr = aspect_attention_layer([sent_context])

    if profile:  # profile unit
        profile_context = Bidirectional(LSTM(dim_lstm,
                                             dropout=dropout,
                                             recurrent_dropout=dropout,
                                             return_state=True,
                                             return_sequences=False),
                                        merge_mode=None,
                                        input_shape=(max_tgtlen, K.int_shape(t_rep)[-1],))
        _, _, p_h_fw, p_c_fw, p_h_bw, p_c_bw = profile_context(p_rep)
        context_encoding_layer_p = Bidirectional(LSTM(dim_lstm,
                                                      unroll=True,
                                                      dropout=dropout,
                                                      recurrent_dropout=dropout,
                                                      return_state=False,
                                                      return_sequences=True),
                                                 merge_mode='concat',
                                                 input_shape=(max_seqlen, K.int_shape(s_rep)[-1],))
        sent_context_p = context_encoding_layer_p(
            s_rep, initial_state=[p_h_fw, p_c_fw, p_h_bw, p_c_bw])
        # Aspect Attention Layer
        aspect_attention_layer_p = AspectAttentionLayer(
            n_reason=num_reason, hidden_d=dim_dense)
        aspect_repr_p = aspect_attention_layer_p([sent_context_p])
        aspect_repr = concatenate([aspect_repr, aspect_repr_p])

    # Prediction layer
    pred = PredictLayer(dim_dense,
                        input_dim=K.int_shape(aspect_repr)[-1],
                        dim_output=dim_output,
                        dropout=dropout)(aspect_repr)

    # Build model graph
    if profile:
        model = Model(inputs=(s, t, p), outputs=pred)
    else:
        model = Model(inputs=(s, t), outputs=pred)

    # Compile model
    nadam = optimizers.Nadam(lr=lr)
    model.compile(
        loss='categorical_crossentropy',  # need one-hot-encoding on y
        # loss='sparse_categorical_crossentropy',  # accept 1-dim y; slow?
        optimizer=nadam,
        metrics=['accuracy']
    )
    logger.debug('...Compile done.')
    # model.summary()
    return model
