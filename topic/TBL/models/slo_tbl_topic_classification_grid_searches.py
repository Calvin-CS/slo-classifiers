"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-23-19

Final Project - SLO TBL Topic Classification

###########################################################
Notes:

Utilizes Scikit-Learn machine learning algorithms for fast prototyping and topic classification using a variety
of Classifiers.

TODO - resolve SettingWithCopyWarning.

TODO - attempt to acquire additional labeled Tweets for topic classification using pattern matching and pandas queries.
TODO - reference settings.py and autocoding.py for template of how to do this.

TODO - revise report.ipynb and paper as updates are made to implementation and code-base.

###########################################################
Resources Used:

Refer to slo_topic_classification_v1-0.py.

https://scikit-plot.readthedocs.io/en/stable/index.html
(visualizations simplified)

"""

################################################################################################################
################################################################################################################

import logging as log
import warnings
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nltk as nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn import metrics

#############################################################

# Note: FIXME - indicates unresolved import error, but still runs fine.
# noinspection PyUnresolvedReferences
from slo_tbl_dataset_preprocessor import tweet_dataset_preprocessor_1, tweet_dataset_preprocessor_2, \
    tweet_dataset_preprocessor_3

#############################################################

# Note: Need to set level AND turn on debug variables in order to see all debug output.
log.basicConfig(level=log.DEBUG)
tf.logging.set_verbosity(tf.logging.ERROR)

# Miscellaneous parameter adjustments for pandas and python.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

"""
Turn debug log statements for various sections of code on/off.
"""
# Debug the GridSearch functions for each Classifier.
debug_pipeline = False
# Debug the initial dataset import and feature/target set creation.
debug_preprocess_tweets = False
# Debug create_training_and_test_set() function.
debug_train_test_set_creation = False


################################################################################################################
################################################################################################################

# Import the datasets.
tweet_dataset_processed1 = \
    pd.read_csv("preprocessed-datasets/tbl_kvlinden_PROCESSED.csv", sep=",")

tweet_dataset_processed2 = \
    pd.read_csv("preprocessed-datasets/tbl_training_set_PROCESSED.csv", sep=",")

# Reindex and shuffle the data randomly.
tweet_dataset_processed1 = tweet_dataset_processed1.reindex(
    pd.np.random.permutation(tweet_dataset_processed1.index))

tweet_dataset_processed2 = tweet_dataset_processed2.reindex(
    pd.np.random.permutation(tweet_dataset_processed2.index))

# Generate a Pandas dataframe.
tweet_dataframe_processed1 = pd.DataFrame(tweet_dataset_processed1)
tweet_dataframe_processed2 = pd.DataFrame(tweet_dataset_processed2)

if debug_preprocess_tweets:
    # Print shape and column names.
    log.debug("\n")
    log.debug("The shape of our SLO dataframe 1:")
    log.debug(tweet_dataframe_processed1.shape)
    log.debug("\n")
    log.debug("The columns of our SLO dataframe 1:")
    log.debug(tweet_dataframe_processed1.head)
    log.debug("\n")
    # Print shape and column names.
    log.debug("\n")
    log.debug("The shape of our SLO dataframe 2:")
    log.debug(tweet_dataframe_processed2.shape)
    log.debug("\n")
    log.debug("The columns of our SLO dataframe 2:")
    log.debug(tweet_dataframe_processed2.head)
    log.debug("\n")

# Concatenate the individual datasets together.
frames = [tweet_dataframe_processed1, tweet_dataframe_processed2]
slo_dataframe_combined = pd.concat(frames, ignore_index=True)

# Reindex everything.
slo_dataframe_combined.index = pd.RangeIndex(len(slo_dataframe_combined.index))
# slo_dataframe_combined.index = range(len(slo_dataframe_combined.index))

# Assign column names.
tweet_dataframe_processed_column_names = ['Tweet', 'SLO']

# Create input features.
selected_features = slo_dataframe_combined[tweet_dataframe_processed_column_names]
processed_features = selected_features.copy()

if debug_preprocess_tweets:
    # Check what we are using as inputs.
    log.debug("\n")
    log.debug("The Tweets in our input feature:")
    log.debug(processed_features['Tweet'])
    log.debug("\n")
    log.debug("SLO TBL topic classification label for each Tweet:")
    log.debug(processed_features['SLO'])
    log.debug("\n")

# Create feature set and target sets.
slo_feature_set = processed_features['Tweet']
slo_target_set = processed_features['SLO']


#######################################################
def create_training_and_test_set():
    """
    This functions splits the feature and target set into training and test sets for each set.

    Note: We use this to generate a randomized training and target set in order to average our results over
    n iterations.

    random_state = rng (where rng = random number seed generator)

    :return: Nothing.  Global variables are established.
    """
    global tweet_train, tweet_test, target_train, target_test, target_train_encoded, target_test_encoded

    from sklearn.model_selection import train_test_split

    import random
    rng = random.randint(1, 1000000)
    # Split feature and target set into training and test sets for each set.
    tweet_train, tweet_test, target_train, target_test = train_test_split(slo_feature_set, slo_target_set,
                                                                          test_size=0.33,
                                                                          random_state=rng)

    if debug_train_test_set_creation:
        log.debug("Shape of tweet training set:")
        log.debug(tweet_train.data.shape)
        log.debug("Shape of tweet test set:")
        log.debug(tweet_test.data.shape)
        log.debug("Shape of target training set:")
        log.debug(target_train.data.shape)
        log.debug("Shape of target test set:")
        log.debug(target_test.data.shape)
        log.debug("\n")

    #######################################################

    # Use Sci-kit learn to encode labels into integer values - one assigned integer value per class.
    from sklearn import preprocessing

    target_label_encoder = preprocessing.LabelEncoder()
    target_train_encoded = target_label_encoder.fit_transform(target_train)
    target_test_encoded = target_label_encoder.fit_transform(target_test)

    target_train_decoded = target_label_encoder.inverse_transform(target_train_encoded)
    target_test_decoded = target_label_encoder.inverse_transform(target_test_encoded)

    if debug_train_test_set_creation:
        log.debug("Encoded target training labels:")
        log.debug(target_train_encoded)
        log.debug("Decoded target training labels:")
        log.debug(target_train_decoded)
        log.debug("\n")
        log.debug("Encoded target test labels:")
        log.debug(target_test_encoded)
        log.debug("Decoded target test labels:")
        log.debug(target_test_decoded)
        log.debug("\n")

    # return [tweet_train, tweet_test, target_train, target_test, target_train_encoded, target_test_encoded]


################################################################################################################
def multinomial_naive_bayes_classifier_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn.naive_bayes import MultinomialNB

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    multinomial_nb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10],
    }

    # Perform the grid search using all cores.
    multinomial_nb_clf = GridSearchCV(multinomial_nb_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    multinomial_nb_clf.fit(tweet_train, target_train)
    multinomial_nb_predictions = multinomial_nb_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(multinomial_nb_clf.cv_results_)
        log.debug("The shape of the Multinomial Naive Bayes Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug("The contents of the Multinomial Naive Bayes Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Multinomial Naive Bayes Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, multinomial_nb_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Multinomial Naive Bayes Classifier Grid Search is: ")
    log.debug(np.mean(multinomial_nb_predictions == target_test))
    log.debug("\n")


################################################################################################################
def sgd_classifier_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn.linear_model import SGDClassifier

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    sgd_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(alpha=0.1, average=False, class_weight=None,
                              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=False,
                              l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=5,
                              n_iter=None, n_iter_no_change=5, n_jobs=-1, penalty='l2',
                              power_t=0.5, random_state=None, shuffle=True, tol=1e-1,
                              validation_fraction=0.1, verbose=0, warm_start=False)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        # 'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-1, 1e-3, 1e-5),
        'clf__fit_intercept': [True, False],
        'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'clf__penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'clf__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'clf__early_stopping': [True, False],
        'clf__validation_fraction': [0.1, 0.2, 0.4],
        'clf__tol': [1e-1, 1e-3, 1e-5],

    }

    # Perform the grid search using all cores.
    sgd_classifier_clf = GridSearchCV(sgd_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    sgd_classifier_clf.fit(tweet_train, target_train)
    sgd_classifier_predictions = sgd_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(sgd_classifier_clf.cv_results_)
        log.debug("The shape of the SGD Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug("The contents of the SGD Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the SGD Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, sgd_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Stochastic Gradient Descent Classifier Grid Search is: ")
    log.debug(np.mean(sgd_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def svm_support_vector_classification_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn import svm

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'clf__gamma': ['scale', 'auto'],
        'clf__shrinking': (True, False),
        'clf__probability': (True, False),
        'clf__tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'clf__decision_function_shape': ['ovo', 'ovr'],
    }

    # Perform the grid search using all cores.
    svc_classifier_clf = GridSearchCV(svc_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    svc_classifier_clf.fit(tweet_train, target_train)
    svc_classifier_predictions = svc_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(svc_classifier_clf.cv_results_)
        log.debug("The shape of the Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug("The contents of the Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Support Vector Classification Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, svc_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Support Vector Classification Classifier Grid Search is: ")
    log.debug(np.mean(svc_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def svm_linear_support_vector_classification_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn import svm

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    linear_svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', svm.LinearSVC(C=0.7, class_weight=None, dual=True, fit_intercept=True,
                              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                              verbose=0)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__penalty': ['l2'],
        'clf__loss': ['squared_hinge'],
        # 'clf__dual': (True, False),
        'clf__multi_class': ['ovr', 'crammer_singer'],
        'clf__tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'clf__fit_intercept': (True, False),
        'clf__max_iter': [500, 1000, 1500, 2000],
    }

    # Perform the grid search using all cores.
    linear_svc_classifier_clf = GridSearchCV(linear_svc_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    linear_svc_classifier_clf.fit(tweet_train, target_train)
    linear_svc_classifier_predictions = linear_svc_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(linear_svc_classifier_clf.cv_results_)
        log.debug("The shape of the Linear Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Linear Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Linear Support Vector Classification Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, linear_svc_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Linear Support Vector Classification Classifier Grid Search is: ")
    log.debug(np.mean(linear_svc_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def nearest_kneighbor_classifier_grid_search():
    """
       Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

       :return: Nothing.
       """
    from sklearn.neighbors import KNeighborsClassifier

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    k_neighbor_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__n_neighbors': [10, 15, 20, 25, 30],
        'clf__weights': ['uniform', 'distance'],
        'clf__algorithm': ['auto'],
        'clf__leaf_size': [5, 10, 15, 20],
        'clf__p': [1, 2, 3, 4],
        'clf__metric': ['euclidean', 'manhattan'],
    }

    # Perform the grid search using all cores.
    k_neighbor_classifier_clf = GridSearchCV(k_neighbor_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    k_neighbor_classifier_clf.fit(tweet_train, target_train)
    k_neighbor_classifier_predictions = k_neighbor_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(k_neighbor_classifier_clf.cv_results_)
        log.debug("The shape of the KNeighbor Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the  KNeighbor Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the KNeighbor Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, k_neighbor_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using KNeighbor Classifier Grid Search is: ")
    log.debug(np.mean(k_neighbor_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def decision_tree_classifier_grid_search():
    """
       Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

       :return: Nothing.
       """
    from sklearn import tree

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    decision_tree_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', tree.DecisionTreeClassifier()),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [None],
        'clf__min_samples_split': [2, 3, 4],
        'clf__min_samples_leaf': [1, 2, 3, 4],
        'clf__min_weight_fraction_leaf': [0],
        'clf__max_features': [None, 'sqrt', 'log2'],
        'clf__max_leaf_nodes': [None, 2, 3, 4],
        'clf__min_impurity_decrease': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    }

    # Perform the grid search using all cores.
    decision_tree_classifier_clf = GridSearchCV(decision_tree_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    decision_tree_classifier_clf.fit(tweet_train, target_train)
    decision_tree_classifier_predictions = decision_tree_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(decision_tree_classifier_clf.cv_results_)
        log.debug("The shape of the Decision Tree Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Decision Tree Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Decision Tree Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, decision_tree_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Decision Tree Classifier Grid Search is: ")
    log.debug(np.mean(decision_tree_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def multi_layer_perceptron_classifier_grid_search():
    """
         Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

         :return: Nothing.
         """
    from sklearn.neural_network import MLPClassifier

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    mlp_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MLPClassifier(activation='logistic', alpha=1e-1, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant', learning_rate_init=1e-1,
                              max_iter=1000, momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='sgd', tol=0.0001,
                              validation_fraction=0.1, verbose=False, warm_start=False)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1)],
        # 'tfidf__use_idf': (True, False),
        # 'clf__hidden_layer_sizes': [(15, 15), (50, 50)],
        'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'clf__solver': ['lbfgs', 'sgd', 'adam'],
        'clf__alpha': [1e-1, 1e-2, 1e-4, 1e-6, 1e-8],
        # 'clf__batch_size': [5, 10, 20, 40, 80, 160],
        'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'clf__learning_rate_init': [1e-1, 1e-3, 1e-5],
        # 'clf__power_t': [0.1, 0.25, 0.5, 0.75, 1.0],
        # 'clf__max_iter': [200, 400, 800, 1600],
        # 'clf_shuffle': [True, False],
        # 'clf__tol': [1e-1, 1e-2, 1e-4, 1e-6, 1e-8],
        # 'clf__momentum': [0.1, 0.3, 0.6, 0.9],
        # 'clf_nestesrovs_momentum': [True, False],
        # 'clf_early_stopping': [True, False],
        # 'clf__validation_fraction': [0.1, 0.2, 0.4],
        # 'clf_beta_1': [0.1, 0.2, 0.4, 0.6, 0.8],
        # 'clf_beta_2': [0.1, 0.2, 0.4, 0.6, 0.8],
        # 'clf_epsilon': [1e-1, 1e-2, 1e-4, 1e-8],
        # 'clf__n_iter_no_change': [1, 2, 4, 8, 16]

    }

    # Perform the grid search using all cores.
    mlp_classifier_clf = GridSearchCV(mlp_classifier_clf, parameters, cv=5, iid=False,
                                      n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    mlp_classifier_clf.fit(tweet_train, target_train)
    mlp_classifier_predictions = mlp_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(mlp_classifier_clf.cv_results_)
        log.debug("The shape of the Multi Layer Perceptron Neural Network Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Multi Layer Perceptron Neural Network Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Multi Layer Perceptron Neural Network Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, mlp_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Multi Layer Perceptron Neural Network Classifier Grid Search is: ")
    log.debug(np.mean(mlp_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def logistic_regression_classifier_grid_search():
    """
       Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

       :return: Nothing.
       """
    from sklearn.linear_model import LogisticRegression

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    logistic_regression_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(random_state=0, solver='lbfgs',
                                   multi_class='multinomial', n_jobs=-1)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__penalty': ['l2'],
        'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'clf__C': [0.2, 0.4, 0.6, 0.8, 1.0],
        'clf__fit_intercept': [True, False],
        'clf__class_weight': ['balanced', None],
        'clf__solver': ['saga', 'newton-cg', 'sag', 'lbfgs'],
        'clf__max_iter': [2000, 4000, 8000, 16000],
        'clf__multi_class': ['ovr', 'multinomial'],
    }

    # Perform the grid search using all cores.
    logistic_regression_classifier_clf = GridSearchCV(logistic_regression_classifier_clf, parameters, cv=5, iid=False,
                                                      n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    logistic_regression_classifier_clf.fit(tweet_train, target_train)
    logistic_regression_classifier_predictions = logistic_regression_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(logistic_regression_classifier_clf.cv_results_)
        log.debug("The shape of the  Logistic Regression Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Logistic Regression Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Logistic Regression Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, logistic_regression_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Logistic Regression Classifier Grid Search is: ")
    log.debug(np.mean(logistic_regression_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################

############################################################################################
"""
Main function.  Execute the program.
"""
import time

if __name__ == '__main__':

    start_time = time.time()

    ################################################
    """
    This section calls grid search functions for automated hyper parameter tuning.
    """
    multinomial_naive_bayes_classifier_grid_search()
    # sgd_classifier_grid_search()
    # svm_support_vector_classification_grid_search()
    # svm_linear_support_vector_classification_grid_search()
    # nearest_kneighbor_classifier_grid_search()
    # decision_tree_classifier_grid_search()
    # multi_layer_perceptron_classifier_grid_search()
    # logistic_regression_classifier_grid_search()

    ################################################

    end_time = time.time()

    if debug_pipeline:
        log.debug("The time taken to train the classifier(s), make predictions, and visualize the results is:")
        total_time = end_time - start_time
        log.debug(str(total_time))
        log.debug("\n")

############################################################################################
