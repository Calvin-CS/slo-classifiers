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

# load the data, which will be split into train and test sets

# make a vectorizers which 

def main(trainset_fp,
         testset_fp,
         output_fp):
#"/home/bjr33/workspace/cs/data/stance/coding/auto_trainset_tok.csv"
    full_training_dataset = pd.read_csv(trainset_fp)

    training_data = full_training_dataset['tweet_t']
    training_labels = full_training_dataset['stance']

    print(training_data.size)
    print(training_labels.size)
#"/home/bjr33/workspace/cs/data/stance/coding/gold_20180514_majority_fixed_tok.csv"
    full_testing_dataset = pd.read_csv(testset_fp)

    testing_data = full_testing_dataset['tweet_t']
    testing_labels = full_testing_dataset['stance']

    vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', use_idf=False)

    vectorized_trainset_data = vectorizer.fit_transform(training_data)
    vectorized_testset_data = vectorizer.transform(testing_data)

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

        scores = cross_val_score(classifier, vectorized_trainset_data, training_labels, cv=5, scoring='f1_weighted')
        f1_means.append(scores.mean())
        f1_stds.append(scores.std())
        print('\tXval F1: {} (+/- {})'.format(scores.mean(), scores.std() * 2))

        classifier.fit(vectorized_trainset_data, training_labels)
        score = f1_score(testing_labels, classifier.predict(vectorized_testset_data), average='weighted')
        print('\tTest F1: {}'.format(score))

        df_f1_scores.loc[i] = [type(classifier).__name__, score]
    with open(output_fp, 'a') as f:
        df_f1_scores.to_csv(f, index=False, header=False)

################################################
# Plot the results.

# _, ax = plt.subplots(figsize=(8, max(len(f1_means), 1.5)))
# y_pos = np.arange(len(f1_means))
# bars = ax.barh(y_pos, f1_means, height=0.5, align='center', xerr=f1_stds)
# ax.set_yticks(y_pos)
# ax.set_yticklabels([type(x).__name__ for x in CLASSIFIERS])
# plt.gca().invert_yaxis()
# ax.set_xlim(0.0, 1.0)
# ax.set_xlabel('Stance XVal F1-Weighted Score')
# ax.set_title('Classifiers for the SemEval 2016 Task A Dataset')

# def autolabel(rects):
#     for rect in rects:
#         width = rect.get_width()
#         ax.text(rect.get_width() + 0.01, rect.get_y() + rect.get_height()/5.0,
#                 '{:.3f}'.format(width),
#                 ha='left', va='center')
# autolabel(bars)

# plt.tight_layout()
# plt.show()

#clf.fit(training_data, training_labels)

#predictions = clf.predict(testing_data)

#print(metrics.classification_report(testing_labels, predictions))

#cm = metrics.confusion_matrix(testing_labels, predictions)
#print(cm)

if __name__ == '__main__':
    Fire(main)
