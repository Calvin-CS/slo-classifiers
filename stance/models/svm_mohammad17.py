"""SVM baseline

`svm_mohammad17.py` is a baseline stance classifier proposed in the
following article:

Stance and Sentiment in Tweets. Saif M. Mohammad, Parinaz Sobhani,
and Svetlana Kiritchenko. Special Section of the ACM Transactions on
Internet Technology on Argumentation in Social Media, 2017, 17(3).

Specification
- Algorithm: Linear SVM
- Features:
    - n-grams:
        - word-wise 1, 2, and 3 tokens
        - char-wise 2, 3, 4, and 5 characters
    - target: presence/absence of the target tokens
    - word embeddings: component-wise averages of all word vectors
      appearing in the tweet
- Hyper parameters: tuned by GridSearchCV
- Preprocessing: tokenised by CMU Tweet Tagger (Gimpel et al, 2011)
"""

# from datetime import datetime
from typing import List

import numpy as np

from data_utility import split_x_value
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer
from sklearn.metrics.classification import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


class TargetVectorizer(BaseEstimator, TransformerMixin):
    """Extract target presence feature from tweets.

    Mohammad '17 says:
        For instance, for 'Hillary Clinton', the mention of either 'Hillary' or 'Clinton' (case insensitive; with or without hashtag) in the tweet shows the presence of target.
    """

    def __init__(self, profile: bool) -> None:
        self.profile = profile

    def get_feature_names(self) -> np.ndarray:
        return np.array(['target'])

    def fit(self, tweets, y=None):
        return self

    def expand_target_words(self, target):
        """Create list of target mention forms, e.g.,
        'Hillary Clinton' -> hillary, clinton, #hillary, #clinton
        """
        targets = [t.lower() for t in target.split()]
        targets.extend(['#' + t.lower() for t in target.split()])
        return targets

    def transform(self, x_values):
        presences = [0] * len(x_values)
        for i, x_value in enumerate(x_values):
            target, tweet, _ = split_x_value(x_value, self.profile)
            targets = self.expand_target_words(target)
            for word in tweet.split():
                if word.lower() in targets:
                    presences[i] = 1
                    break
        return np.array([presences]).T


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Get tweet representation from pre-trained word vectors."""

    def __init__(self, wordvec: KeyedVectors, profile: bool) -> None:
        self.wordvec = wordvec
        self.wordvec_dim = self.wordvec.vector_size
        self.zeros = np.zeros(self.wordvec_dim)
        self.profile = profile

    def get_feature_names(self) -> np.ndarray:
        return np.array([f'd{i}' for i in range(1, self.wordvec_dim + 1)])

    def fit(self, tweets, y=None):
        return self

    def transform(self, x_values):
        ret = []
        for x_value in x_values:
            # Include embeddings for target, tweet and profile texts.
            target, tweet, profile = split_x_value(x_value, self.profile)
            words = target.split(' ') + tweet.split(' ') + profile.split(' ')
            ret.append(np.array([
                self.wordvec[word] if word in self.wordvec else self.zeros
                for word in words
            ]).mean(axis=0))

        return np.array(ret)


class SLO_WordAnalyzer(BaseEstimator):
    def __init__(self, profile: bool) -> None:
        self.profile = profile

    def __call__(self, doc: str) -> List[str]:
        if self.profile:
            # Concatenate target, tweet text and profile description,
            # prefixing target and profile tokens to distinguish them
            # from the tweet text.
            target, text, desc = split_x_value(doc, self.profile)
            target_tokens = [f't_{tok}' for tok in target.split()]
            t_tokens = text.split()
            p_tokens = [f'p_{tok}' for tok in desc.split()]
            return target_tokens + t_tokens + p_tokens
        else:
            # regard every character including signs as tokens
            # sklearn's word analyzer only consider usual words and ignore all punctuations and symbols including 1 character word
            # we can explicitly remove non-word characters or stopwords here
            return doc.split()


def get_model(wordvec: KeyedVectors,
              profile: bool=False) -> GridSearchCV:

    # slo_word_analyzer = SLO_WordAnalyzer(profile)
    slo_word_analyzer = SLO_WordAnalyzer(profile)
    word_ngram = CountVectorizer(
        # analyzer='word',  # we include symbols
        analyzer=slo_word_analyzer,
        binary=True,
        ngram_range=(1, 3),
        lowercase=False
    )
    char_ngram = CountVectorizer(
        analyzer='char',
        binary=True,
        ngram_range=(2, 5),
        lowercase=False
    )
    tv = TargetVectorizer(profile)
    ev = EmbeddingVectorizer(wordvec, profile)

    features = FeatureUnion([
        ('ngram_w', word_ngram),
        ('ngram_c', char_ngram),
        ('target', tv),
        ('embedding', ev)
    ])

    svm = LinearSVC(C=3.0)

    # return Pipeline([('vect', features), ('clf', svm)])
    model = Pipeline([('vect', features), ('clf', svm)])

    macro_f = make_scorer(f1_score, labels=[0, 1, 2], average='macro')
    searcher = GridSearchCV(
        model,
        param_grid=dict(
            # clf__C=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
            clf__C=[1, 3, 6, 9],
            # clf__penalty=['l1', 'l2'],
            # clf__class_weight=['', 'balanced']
        ),
        cv=5,
        scoring=macro_f,
        n_jobs=-1,
        # verbose=1,
        # return_train_score=True
    )

    return searcher


# not used
def train_cv(model: Pipeline, x_train: np.ndarray, y_train: np.ndarray, cv: int =5) -> Pipeline:
    macro_f = make_scorer(f1_score, labels=[0, 1], average='macro')
    searcher = GridSearchCV(
        model,
        param_grid=dict(
            # clf__C=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
            clf__C=[1, 10, 100, 1000]
            # clf__penalty=['l1', 'l2'],
            # clf__class_weight=['', 'balanced']
        ),
        cv=cv,
        scoring=macro_f,
        n_jobs=-1,
        # verbose=1,
        # return_train_score=True
    )
    searcher.fit(x_train, y_train)
    # print('best SVM C:'.rjust(45), searcher.best_params_['clf__C'])
    # pd.DataFrame(searcher.cv_results_).to_csv(f"train{cv}cv_results-{datetime.today():%Y%m%dT%H%M}.csv")

    return searcher.best_estimator_
