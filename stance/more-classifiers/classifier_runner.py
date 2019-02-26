import os
import sys
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from fire import Fire

def main(trainset_fp,
         testset_fp,
         output_fp):
    # load the data, which will be split into train and test sets
    full_training_dataset = pd.read_csv(trainset_fp)

    training_data = full_training_dataset['tweet_t']
    training_labels = full_training_dataset['stance']

    full_testing_dataset = pd.read_csv(testset_fp)

    testing_data = full_testing_dataset['tweet_t']
    testing_labels = full_testing_dataset['stance']

    # Set the vectorizer to transform the data into inputs for classifiers
    vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', use_idf=False)

    vectorized_trainset_data = vectorizer.fit_transform(training_data)
    vectorized_testset_data = vectorizer.transform(testing_data)

    #Specify the classifiers to be used
    CLASSIFIERS = [
        DummyClassifier(strategy='most_frequent'),  # Stratified works better than most_frequent (used for SemEval).
        BernoulliNB(),
        MultinomialNB(),
        LogisticRegression(C=1e5),
        SGDClassifier(max_iter=5, tol=None),
        LinearSVC(),
        MLPClassifier(),
        SVC(),
        RandomForestClassifier(),
        PassiveAggressiveClassifier(),
    ]

    df_f1_scores = pd.DataFrame(columns=['classifier', 'f1_score'])

    f1_means = []
    f1_stds = []

    for i, classifier in enumerate(CLASSIFIERS):
        print(type(classifier).__name__)

        # Do cross validation on the training set
        scores = cross_val_score(classifier, vectorized_trainset_data, training_labels, cv=5, scoring='f1_weighted')
        f1_means.append(scores.mean())
        f1_stds.append(scores.std())
        print('\tXval F1: {} (+/- {})'.format(scores.mean(), scores.std() * 2))

        # Fit the data to the classifier
        classifier.fit(vectorized_trainset_data, training_labels)
        # Test the fit of the classifier using the testing data
        score = f1_score(testing_labels, classifier.predict(vectorized_testset_data), average='weighted')
        print('\tTest F1: {}'.format(score))
        df_f1_scores.loc[i] = [type(classifier).__name__, score]

    # Append score to output csv file
    with open(output_fp, 'a') as f:
        df_f1_scores.to_csv(f, index=False, header=False)

if __name__ == '__main__':
    Fire(main)
