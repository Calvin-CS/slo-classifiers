'''Memory network for stance detection.

Modification of keras/examples/babi-memnn.py

References:

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

- Cheng Li, Xiaoxiao Guo, Qiaozhu Mei,
  "Deep Memory Networks for Attitude Identification"
  http://arxiv.org/abs/1701.04189
'''
import logging

import keras.backend as K
from keras import optimizers
from keras.layers import (LSTM, Activation, Dense, Dropout, Input, Permute,
                          add, concatenate, dot)
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential

logger = logging.getLogger(__name__)


def memory_layer(input_sequence, vocab_size,
                 question_encoded,
                 dropout=0.3,
                 input_encoded_m=None,
                 input_encoded_c=None,
                 embedding_matrix=None,
                 input_profile=None,
                 input_encoded_p=None,
                 input_encoded_d=None):
    _, max_tgtlen, dim_wordvec = K.int_shape(question_encoded)
    max_seqlen = K.int_shape(input_sequence)[-1]
    if input_profile is not None:
        max_prflen = K.int_shape(input_profile)

    # encoders
    if input_encoded_m is None and input_encoded_c is None:
        # embed the input sequence into a sequence of vectors
        input_encoder_m = generate_encoder(
            vocab_size, embedding_matrix, dim_wordvec, max_seqlen, dropout)
        # output: (samples, max_seqlen, embedding_dim)
        # embed the input into a sequence of vectors of size max_tgtlen
        # input_encoder_c = generate_encoder(
        #     vocab_size, None, max_tgtlen, None, dropout)
        # output: (samples, max_seqlen, max_tgtlen)
        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        # original C matrix
        # input_encoded_c = input_encoder_c(input_sequence)
        # Transformer style KV pairs (samples, max_seqlen, dim_wordvec)
        input_encoded_c = input_encoded_m
    if input_profile is not None:
        if input_encoded_p is None and input_encoded_d is None:
            input_encoder_p = generate_encoder(
                vocab_size, embedding_matrix, dim_wordvec, max_prflen, dropout)
            input_encoder_d = generate_encoder(
                vocab_size, None, max_tgtlen, None, dropout)
            input_encoded_p = input_encoder_p(input_profile)
            input_encoded_d = input_encoder_d(input_profile)
            # Transformer style KV pairs (samples, max_seqlen, dim_wordvec)
            input_encoded_d = input_encoded_p

    # compute a 'match' between the first input vector sequence
    # and the question vector sequence
    # shape: `(samples, max_seqlen, max_tgtlen)`
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)
    if input_profile is not None:
        # shape: `(samples, max_prflen, max_tgtlen)`
        match_p = dot([input_encoded_p, question_encoded], axes=(2, 2))
        match_p = Activation('softmax')(match_p)

    # add the match matrix with the second input vector sequence
    # (samples, max_seqlen, max_tgtlen)
    # response = add([match, input_encoded_c])
    # response = Permute((2, 1))(response)  # (samples, max_tgtlen, max_seqlen)
    # TODO: ScaledDotProductAttention?
    response = dot([match, input_encoded_c], axes=(1, 1))
    # shape: `(samples, max_tgtlen, dim_wordvec)`
    if input_profile is not None:
        # (samples, max_prflen, max_tgtlen)
        # response_p = add([match_p, input_encoded_d])
        # (samples, max_tgtlen, max_prflen)
        # response_p = Permute((2, 1))(response_p)
        response_p = dot([match_p, input_encoded_d], axes=(1, 1))

    # concatenate the match matrix with the question vector sequence
    # answer = concatenate([response, question_encoded])
    answer = add([response, question_encoded])
    # (samples, max_tgtlen, embedding_dim)
    if input_profile is not None:
        answer = concatenate([answer, response_p])
        # (samples, max_tgtlen, embedding_dim*2)

    return answer, input_encoded_m, input_encoded_c


def generate_encoder(vocab_size, embedding_matrix, dim_wordvec, max_len, dropout):
    input_encoder = Sequential()
    if embedding_matrix is not None:
        input_encoder.add(Embedding(input_dim=vocab_size,
                                    weights=[embedding_matrix],
                                    output_dim=dim_wordvec,
                                    input_length=max_len,
                                    trainable=False))
    else:
        input_encoder.add(Embedding(input_dim=vocab_size,
                                    output_dim=dim_wordvec))
    input_encoder.add(Dropout(dropout))
    return input_encoder


def build_model(embedding_matrix=None,
                word_index=None, max_vocabsize=200_000,
                max_seqlen=20, max_tgtlen=4, dim_wordvec=64,
                profile=False, max_prflen=20,
                dropout=0.3, dim_output=3, lr=0.001,
                dim_lstm=32, num_layers=3,
                weight_tying=False, **kwargs):
    """The builder function to compile MemNet.

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

    Args (MemNet Specific):
        dim_lstm: the output dimensionality of LSTM to reduce the match matrix
        num_layers: the number of stacks of the memory layer
        weight_tying: whether to share embedding weights in all layers (layer-wise tying)

    Returns:
        compiled keras.models.Model
    """
    logger.debug('Compiling MemNet...')

    vocab_size = min(max_vocabsize, len(word_index)) + 1
    if embedding_matrix is not None:
        dim_wordvec = embedding_matrix.shape[1]

    # placeholders
    input_sequence = Input((max_seqlen,))
    question = Input((max_tgtlen,))
    if profile:
        input_profile = Input((max_prflen,))
    else:
        input_profile, input_encoded_p, input_encoded_d = None, None, None

    # embed the question into a sequence of vectors
    question_encoder = generate_encoder(
        vocab_size, embedding_matrix, dim_wordvec, max_tgtlen, dropout)
    # output: (samples, max_tgtlen, embedding_dim)

    # initialise for loop
    answer = question_encoder(question)
    input_encoded_m, input_encoded_c = None, None
    # multiple layers
    for _ in range(num_layers):
        if weight_tying:
            answer, input_encoded_m, input_encoded_c = memory_layer(
                input_sequence, vocab_size, answer,
                dropout=dropout,
                input_encoded_m=input_encoded_m,
                input_encoded_c=input_encoded_c,
                embedding_matrix=embedding_matrix,
                input_profile=input_profile,
                input_encoded_p=input_encoded_p,
                input_encoded_d=input_encoded_d)
        else:
            answer, _, _ = memory_layer(
                input_sequence, vocab_size, answer,
                dropout=dropout,
                embedding_matrix=embedding_matrix,
                input_profile=input_profile)
        answer = TimeDistributed(Dense(dim_wordvec))(answer)

    # the original paper uses a matrix multiplication for this reduction step.
    # we choose to use a RNN instead.
    # TODO: provide options for this preditction step [lstm, cnn, dense]
    answer = LSTM(dim_lstm)(answer)  # (samples, 32)

    # one regularization layer -- more would probably be needed.
    answer = Dropout(dropout)(answer)
    answer = Dense(dim_output)(answer)  # (samples, vocab_size)
    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)

    # build the final model
    nadam = optimizers.Nadam(lr=lr)
    if profile:
        model = Model([input_sequence, question, input_profile], answer)
    else:
        model = Model([input_sequence, question], answer)
    model.compile(
        optimizer=nadam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # model.summary()
    logger.debug('...Compile done.')
    return model

    # # train
    # model.fit([inputs_train, queries_train], answers_train,
    # batch_size=32,
    # epochs=120,
    # validation_data=([inputs_test, queries_test], answers_test))
