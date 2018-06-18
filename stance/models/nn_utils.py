"""Sklearn compatible wrapper for keras models of SLO stance detection."""
import logging

import numpy as np

import keras.backend as K
from keras.callbacks import EarlyStopping  # , ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

logger = logging.getLogger(__name__)


class TextSeqTransformer(BaseEstimator, TransformerMixin):
    """Transform input text data to sequences of word indices."""

    def __init__(self, wordvec,
                 max_seqlen=20, max_tgtlen=4, max_vocabsize=200_000,
                 profile=False, max_prflen=20, prf_cat=True):
        self.wordvec = wordvec
        self.max_seqlen = max_seqlen
        self.max_vocabsize = max_vocabsize
        self.max_tgtlen = max_tgtlen
        self.profile = profile
        self.max_prflen = max_prflen
        self.prf_cat = prf_cat

        if self.prf_cat:
            logger.debug(
                'two inputs mode -- profile is concatenated to tweet text')
        else:
            logger.debug('three inputs mode -- tweet, target, profile')

    def fit(self, X, y=None):
        logger.debug(f'TextSeqTransformer fit...')
        # X is [x_train, x_test]
        # each row of x is '\t'-joined [target, tweet, profile (if profile)]
        X = np.concatenate(X)
        self.tk = Tokenizer(num_words=self.max_vocabsize)
        # TODO: add parameters of tokenise filter
        # NOTE: keras Tokenizer removes all punctuation, tab, and line breaks. This behaviour is controlled by `filter` option.
        # as default, you don't have to remove '\t' from x here
        self.tk.fit_on_texts(list(X))
        logger.debug('Vocabulary size: %d' % len(self.tk.word_index))
        if self.wordvec:
            self._embed_word_matrix()
        else:
            self.embedding_matrix = None

        return self

    def transform(self, X):
        x_train, x_test = X
        logger.debug(
            f'TextSeqTransformer transform to x_train={x_train.shape}, x_test={x_test.shape}')
        # training text sequence (input matrix: shape - [sentence_len, MAX_SEQUENCE_LENGTH])

        return self._text2sequence(x_train), self._text2sequence(x_test)

    def _text2sequence(self, x):
        X = np.char.split(x, sep='\t')
        X = np.char.array(X.tolist())
        if self.profile:
            assert X.shape == (len(x), 3)
            xt, x, xp = np.hsplit(X, 3)
            X = self.tk.texts_to_sequences(x.flatten())
            X = pad_sequences(X, maxlen=self.max_seqlen)
            Xt = self.tk.texts_to_sequences(xt.flatten())
            Xt = pad_sequences(Xt, maxlen=self.max_tgtlen)
            Xp = self.tk.texts_to_sequences(xp.flatten())
            Xp = pad_sequences(Xt, maxlen=self.max_prflen)
            if self.prf_cat:
                # concatenate profile to text
                X = np.concatenate((X, Xp), axis=1)
                return [X, Xt]
            else:
                # strict separation
                return [X, Xt, Xp]
        else:
            assert X.shape == (len(x), 2)
            xt, x = np.hsplit(X, 2)
            X = self.tk.texts_to_sequences(x.flatten())
            X = pad_sequences(X, maxlen=self.max_seqlen)
            Xt = self.tk.texts_to_sequences(xt.flatten())
            Xt = pad_sequences(Xt, maxlen=self.max_tgtlen)
            return [X, Xt]

    def _embed_word_matrix(self):
        nb_words = min(self.max_vocabsize, len(self.tk.word_index)) + 1
        dim_wordvec = self.wordvec.vector_size
        self.embedding_matrix = np.zeros((nb_words, dim_wordvec))
        for word, i in self.tk.word_index.items():
            if word in self.wordvec:
                self.embedding_matrix[i] = self.wordvec[word]
            else:
                self.embedding_matrix[i] = np.random.rand(dim_wordvec)

        # print('Valid word embeddings: %d' % np.sum(np.sum(self.embedding_matrix, axis=1) != 0))

        # print('saving glove matrix: %s ...' % output_file)
        # np.save(output_file, embedding_matrix)
        # print('saved.')


class NeuralPipeline(BaseEstimator, ClassifierMixin):
    """Ad-hoc wrapper to a keras model for sklearn."""

    def __init__(self, build_fn, wordvec=None,
                 max_vocabsize=200_000,
                 max_seqlen=20, max_tgtlen=4, dim_wordvec=64,
                 profile=False, max_prflen=20, prf_cat=True,
                 target=False, parallel=False, m_profile=1,
                 xtra_self_att=True,
                 dropout=0.1, lr=0.001,
                 dim_lstm=100, num_reason=1, dim_dense=300,
                 weight_tying=False,
                 dim_pff=None, num_head=4, num_layers=2,
                 validation_split=0.2, epochs=100,
                 batch_size=128, patience=20,
                 **kwargs):
        # preprocessing parameters
        self.max_vocabsize = max_vocabsize
        self.max_seqlen = max_seqlen
        self.max_tgtlen = max_tgtlen
        self.profile = profile
        self.max_prflen = max_prflen
        self.prf_cat = prf_cat
        self.vect = TextSeqTransformer(
            wordvec=wordvec,
            max_seqlen=self.max_seqlen,
            max_tgtlen=self.max_tgtlen,
            max_vocabsize=self.max_vocabsize,
            profile=self.profile,
            max_prflen=self.max_prflen,
            prf_cat=self.prf_cat
        )
        self.build_fn = build_fn

        # model parameters
        self.max_vocabsize = max_vocabsize
        self.dropout = dropout
        self.lr = lr
        self.dim_wordvec = dim_wordvec
        self.dim_lstm = dim_lstm
        self.num_reason = num_reason
        self.dim_dense = dim_dense
        self.weight_tying = weight_tying
        self.target = target
        self.parallel = parallel
        self.xtra_self_att = xtra_self_att
        self.m_profile = m_profile
        self.dim_pff = dim_pff
        self.num_head = num_head
        self.num_layers = num_layers

        # train parameters
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.kwargs = kwargs

        self._verbose = max(logger.getEffectiveLevel() // 10, 2)
        # keras verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
        # max(verbose) 2 for fit, 1 for predict
        # 2 is briefer than 1, so this corresponds to logging_level

    def fit(self, X, y=None):
        inputs_train, self.inputs_test = self.vect.fit_transform(X)

        # TODO: parametrise early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience
        )

        if self.prf_cat:
            seqlen = self.max_seqlen + self.max_prflen
        else:
            seqlen = self.max_seqlen

        self.model = self.build_fn(
            embedding_matrix=self.vect.embedding_matrix,
            word_index=self.vect.tk.word_index,
            dim_output=y.shape[1],  # assume input is to_categorical-ed
            max_vocabsize=self.max_vocabsize,
            max_seqlen=seqlen,
            max_tgtlen=self.max_tgtlen,
            dim_wordvec=self.dim_wordvec,
            profile=False if self.prf_cat else self.profile,
            max_prflen=self.max_prflen,
            dropout=self.dropout,
            lr=self.lr,
            dim_lstm=self.dim_lstm,
            num_reason=self.num_reason,
            dim_dense=self.dim_dense,
            target=self.target,
            parallel=self.parallel,
            xtra_self_att=self.xtra_self_att,
            m_profile=self.m_profile,
            dim_pff=self.dim_pff,
            num_head=self.num_head,
            num_layers=self.num_layers,
            weight_tying=self.weight_tying
        )

        logger.debug('fit on data ' + str([inp.shape for inp in inputs_train]))
        self.model.fit(
            x=inputs_train, y=y,
            callbacks=[early_stopping],  # , model_checkpoint],
            validation_split=self.validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True, verbose=self._verbose)

        return self.model

    def predict(self, X):
        # X = self.vect.transform(X)
        logger.debug('predict on data ' +
                     str([inp.shape for inp in self.inputs_test]))
        y_pred = self.model.predict(self.inputs_test,
                                    batch_size=self.batch_size // 2,
                                    verbose=max(self._verbose, 1))
        K.clear_session()
        # fix multiclass one-hot output shape(len(x), len(label)) to 1-dim integer labels
        y_pred = np.array([np.argmax(row) for row in y_pred])

        return y_pred
