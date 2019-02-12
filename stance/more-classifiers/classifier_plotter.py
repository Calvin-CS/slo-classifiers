import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
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

from fire import Fire

def main(results_csv):
    ###############################################
    #Plot the results.

    CLASSIFIERS = [
        BernoulliNB(),
        DummyClassifier(strategy='most_frequent'),  # Stratified works better than most_frequent (used for SemEval).
        LinearSVC(),
        LogisticRegression(C=1e5),
        MLPClassifier(),
        MultinomialNB(),
#	PassiveAggresiveClassifier(),
	RandomForestClassifier(),
        SGDClassifier(max_iter=5, tol=None),
	SVC()
    ]

    df_results = pd.read_csv(results_csv)

    f1_means = df_results['f1_score']
    f1_stds = df_results['f1_sd']

    _, ax = plt.subplots(figsize=(8, max(len(f1_means), 1.5)))
    y_pos = np.arange(len(f1_means))
    bars = ax.barh(y_pos, f1_means, height=0.5, align='center', xerr=f1_stds)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([type(x).__name__ for x in CLASSIFIERS])
    plt.gca().invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Stance F1-Weighted Score')
    ax.set_title('Classifiers using the Jan2010-Jul2018 Dataset')

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.text(rect.get_width() + 0.01, rect.get_y() + rect.get_height()/5.0,
                    '{:.3f}'.format(width),
                    ha='left', va='center')
    autolabel(bars)

    plt.tight_layout()
    #plt.show()
    plt.savefig("results.png")

if __name__ == '__main__':
    Fire(main)
