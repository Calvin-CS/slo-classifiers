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
# Debug the scikit_learn_multinomialnb_classifier_non_pipeline() function.
debug_MNB_nonPipeline = False
# Debug the initial dataset import and feature/target set creation.
debug_preprocess_tweets = False
# Debug create_training_and_test_set() function.
debug_train_test_set_creation = False
# Debug each iteration of training and predictions for each Classifier function().
debug_classifier_iterations = False
# Debug the create_prediction_set() function.
debug_create_prediction_set = False
# Debug the make_predictions() function.
debug_make_predictions = False

"""
Controls the # of iterations to run each Classifier before outputting the mean accuracy metric obtained.

IMPORTANT NOTE: SET to "1" unless you want each Classifier spitting out visualizations for each of N iterations
when visualizations are enabled!
"""
iterations = 1000

"""
Enable or disable making predictions using trained model on our large 650k+ Tweet dataset (takes a little while).
"""
enable_predictions = True

"""
Enable or disable plotting graph visualizations of the model's training and prediction results.
"""
enable_visualizations = False

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


#######################################################
def scikit_learn_multinomialnb_classifier_non_pipeline():
    """
    Function trains a Multinomial Naive Bayes Classifier without using a Pipeline.

    Note: Implemented for educational purposes - so I can see the manual workflow, otherwise the Pipeline Class hides
    these details and we only have to tune parameters.  Also, so I can see how large the bag-of-words vocabulary size
    becomes.

    :return: none.
    """

    # Create the training and test sets from the feature and target sets.
    create_training_and_test_set()

    # Use Sci-kit learn to tokenize each Tweet and convert into a bag-of-words sparse feature vector.
    vectorizer = CountVectorizer(min_df=0, lowercase=False, ngram_range=(1, 1))
    tweet_train_encoded = vectorizer.fit_transform(tweet_train)
    tweet_test_encoded = vectorizer.transform(tweet_test)

    if debug_MNB_nonPipeline:
        log.debug("Vectorized tweet training set:")
        log.debug(tweet_train_encoded)
        log.debug("Vectorized tweet testing set:")
        log.debug(tweet_test_encoded)
        log.debug("Shape of the tweet training set:")
        log.debug(tweet_train_encoded.shape)
        log.debug("Shape of the tweet testing set:")
        log.debug(tweet_test_encoded.shape)

    #######################################################

    # Use Sci-kit learn to convert each tokenized Tweet into term frequencies.
    tfidf_transformer = TfidfTransformer(use_idf=False)

    tweet_train_encoded_tfidf = tfidf_transformer.fit_transform(tweet_train_encoded)
    tweet_test_encoded_tfidf = tfidf_transformer.transform(tweet_test_encoded)

    if debug_MNB_nonPipeline:
        log.debug("vectorized tweet training set term frequencies down-sampled:")
        log.debug(tweet_train_encoded_tfidf)
        log.debug("Shape of the tweet training set term frequencies down-sampled: ")
        log.debug(tweet_train_encoded_tfidf.shape)
        log.debug("\n")
        log.debug("vectorized tweet test set term frequencies down-sampled:")
        log.debug(tweet_test_encoded_tfidf)
        log.debug("Shape of the tweet test set term frequencies down-sampled: ")
        log.debug(tweet_test_encoded_tfidf.shape)
        log.debug("\n")

    #######################################################

    from sklearn.naive_bayes import MultinomialNB

    # Train the Multinomial Naive Bayes Classifier.
    clf_multinomial_nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf_multinomial_nb.fit(tweet_train_encoded_tfidf, target_train_encoded)

    # Predict using the Multinomial Naive Bayes Classifier.
    clf_multinomial_nb_predict = clf_multinomial_nb.predict(tweet_test_encoded_tfidf)

    # FIXME - still working, but issue with string conversion of accuracy metric using accuracy function.
    from sklearn.metrics import accuracy_score
    accuracy_is = accuracy_score(target_test_encoded, clf_multinomial_nb_predict, normalize=True)
    log.debug("MultinomialNB Classifier accuracy using accuracy_score() function : ", str(accuracy_is))
    log.debug("\n")

    # Another method of obtaining accuracy metric.
    log.debug("Accuracy for test set predictions using multinomialNB:")
    log.debug(str(np.mean(clf_multinomial_nb_predict == target_test_encoded)))
    log.debug("\n")

    # View the results as Tweet => predicted topic classification label.
    for index, label in zip(tweet_test, clf_multinomial_nb_predict):
        log.debug('%r => %s' % (index, label))


################################################################################################################
def create_prediction_set():
    """
    Function prepares the borg-classifier dataset to be used for predictions in trained models.

    :return: the prepared dataset.
    """

    # Import the dataset.
    slo_dataset_cmu = \
        pd.read_csv("preprocessed-datasets/dataset_20100101-20180510_tok_PROCESSED.csv", sep=",")

    # Shuffle the data randomly.
    slo_dataset_cmu = slo_dataset_cmu.reindex(
        pd.np.random.permutation(slo_dataset_cmu.index))

    # Generate a Pandas dataframe.
    slo_dataframe_cmu = pd.DataFrame(slo_dataset_cmu)

    if debug_create_prediction_set:
        # Print shape and column names.
        log.debug("\n")
        log.debug("The shape of our SLO CMU dataframe:")
        log.debug(slo_dataframe_cmu.shape)
        log.debug("\n")
        log.debug("The columns of our SLO CMU dataframe:")
        log.debug(slo_dataframe_cmu.head)
        log.debug("\n")

    # Reindex everything.
    slo_dataframe_cmu.index = pd.RangeIndex(len(slo_dataframe_cmu.index))
    # slo_dataframe_cmu.index = range(len(slo_dataframe_cmu.index))

    # Create input features.
    # Note: using "filter()" - other methods seems to result in shape of (658982, ) instead of (658982, 1)
    selected_features_cmu = slo_dataframe_cmu.filter(['tweet_t'])
    processed_features_cmu = selected_features_cmu.copy()

    # Rename column.
    processed_features_cmu.columns = ['Tweets']

    if debug_create_prediction_set:
        # Print shape and column names.
        log.debug("\n")
        log.debug("The shape of our processed features:")
        log.debug(processed_features_cmu.shape)
        log.debug("\n")
        log.debug("The columns of our processed features:")
        log.debug(processed_features_cmu.head)
        log.debug("\n")

    if debug_create_prediction_set:
        # Check what we are using as inputs.
        log.debug("\n")
        log.debug("The Tweets in our input feature:")
        log.debug(processed_features_cmu['Tweets'])
        log.debug("\n")

    return processed_features_cmu


################################################################################################################
def make_predictions(trained_model):
    """
    Function makes predictions using the trained model passed as an argument.

    :param trained_model.
    :return: Nothing.
    """

    # Generate the dataset to be used for predictions.
    prediction_set = create_prediction_set()

    # Make predictions of the borg-slo-classifiers dataset.
    # Note to self: don't be an idiot and try to make predictions on the entire dataframe object instead of a column.
    predictions = trained_model.predict(prediction_set['Tweets'])

    # Store predictions in Pandas dataframe.
    results_df = pd.DataFrame(predictions)

    # Assign column names.
    results_df_column_name = ['TBL_classification']
    results_df.columns = results_df_column_name

    if debug_make_predictions:
        log.debug("The shape of our prediction results dataframe:")
        log.debug(results_df.shape)
        log.debug("\n")
        log.debug("The contents of our prediction results dataframe:")
        log.debug(results_df.head())
        log.debug("\n")

    # Count # of each classifications made.
    social_counter = 0
    economic_counter = 0
    environmental_counter = 0

    for index in results_df.index:
        if results_df['TBL_classification'][index] == 'economic':
            economic_counter += 1
        if results_df['TBL_classification'][index] == 'social':
            social_counter += 1
        if results_df['TBL_classification'][index] == 'environmental':
            environmental_counter += 1

    # Calculate percentages for each classification.
    social_percentage = (social_counter / results_df.shape[0]) * 100.0
    economic_percentage = (economic_counter / results_df.shape[0]) * 100.0
    environmental_percentage = (environmental_counter / results_df.shape[0]) * 100.0

    # Display our statistics.
    log.debug("The number of Tweets identified as social is :" + str(social_counter))
    log.debug("The % of Tweets identified as social in the entire dataset is: " + str(social_percentage))
    log.debug("The number of Tweets identified as economic is :" + str(economic_counter))
    log.debug("The % of Tweets identified as economic in the entire dataset is: " + str(economic_percentage))
    log.debug("The number of Tweets identified as environmental is :" + str(environmental_counter))
    log.debug("The % of Tweets identified as environmental in the entire dataset is: " + str(environmental_percentage))
    log.debug("\n")


################################################################################################################
def create_metric_visualizations(model, model_predictions, model_predictions_probabilities, classifier_name):
    """
    This function visualizes the metrics for a single iteration of predictions.

    :param classifier_name: string name of the classifier we are using.
    :param model: the Scikit-Learn Classifier we are using.
    :param model_predictions: the predictions made using the trained model.
    :param model_predictions_probabilities: the predictions' probabilities made using the trained model.
    :return: Nothing.
    """

    import scikitplot as skplt

    # Plot the confusion matrix.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_confusion_matrix(target_test, model_predictions, normalize=True,
                                        title=str(classifier_name) + ' Confusion Matrix')
    plt.show()

    # Plot the ROC curve.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_roc(target_test, model_predictions_probabilities,
                           title=str(classifier_name) + ' ROC Curves')
    plt.show()

    # Plot the precision and recall curve.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_precision_recall(target_test, model_predictions_probabilities,
                                        title=str(classifier_name) + ' Precision-Recall Curve')
    plt.show()

    # Plot learning curve.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.estimators.plot_learning_curve(model, tweet_train, target_train,
                                         title=str(classifier_name) + ' Learning Curve')
    plt.show()


################################################################################################################
def create_metric_visualizations_linearsvc(model, model_predictions, classifier_name):
    """
    This function visualizes the metrics for a single iteration of predictions.

    Note: Specific to LinearSVC Classifier (generalized version is not compatible)

    :param classifier_name: string name of the classifier we are using.
    :param model: the Scikit-Learn Classifier we are using.
    :param model_predictions: the predictions made using the trained model.
    :return: Nothing.
    """

    import scikitplot as skplt

    # Plot the confusion matrix.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_confusion_matrix(target_test, model_predictions, normalize=True,
                                        title=str(classifier_name) + ' Confusion Matrix')
    plt.show()

    # Plot learning curve.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.estimators.plot_learning_curve(model, tweet_train, target_train,
                                         title=str(classifier_name) + ' Learning Curve')
    plt.show()


################################################################################################################
def create_metric_visualizations_kneighbor(model, model_predictions, model_predictions_probabilities, classifier_name):
    """
    This function visualizes the metrics for a single iteration of predictions.

    Note: Specific to KNeighbors Classifier (generalized version is not compatible)

    :param classifier_name: string name of the classifier we are using.
    :param model: the Scikit-Learn Classifier we are using.
    :param model_predictions: the predictions made using the trained model.
    :param model_predictions_probabilities: the predictions' probabilities made using the trained model.
    :return: Nothing.
    """

    import scikitplot as skplt

    # Plot the confusion matrix.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_confusion_matrix(target_test, model_predictions, normalize=True,
                                        title=str(classifier_name) + ' Confusion Matrix')
    plt.show()

    # Plot the ROC curve.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_roc(target_test, model_predictions_probabilities,
                           title=str(classifier_name) + ' ROC Curves')
    plt.show()

    # Plot the precision and recall curve.
    plt.figure(figsize=(16, 9), dpi=360)
    skplt.metrics.plot_precision_recall(target_test, model_predictions_probabilities,
                                        title=str(classifier_name) + ' Precision-Recall Curve')
    plt.show()

    # # Plot learning curve.
    # plt.figure(figsize=(16, 9), dpi=360)
    # # plt.figure(figsize=(3, 2), dpi=300)
    # skplt.estimators.plot_learning_curve(model, tweet_train, target_train,
    #                                      title=str(classifier_name) + ' Learning Curve')
    # plt.show()


################################################################################################################
def multinomial_naive_bayes_classifier():
    """
    Functions trains a Multinomial Naive Bayes Classifier.

    :return: none.
    """
    from sklearn.naive_bayes import MultinomialNB

    multinomial_nb_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(multinomial_nb_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Multinomial Naive Bayes: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):

        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        multinomial_nb_clf.fit(tweet_train, target_train)
        multinomial_nb_predictions = multinomial_nb_clf.predict(tweet_test)
        multinomial_nb_predictions_probabilities = multinomial_nb_clf.predict_proba(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations(multinomial_nb_clf, multinomial_nb_predictions,
                                         multinomial_nb_predictions_probabilities, 'Multinomial Naive Bayes Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(multinomial_nb_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Multinomial Naive Bayes Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("Multinomial Naive Bayes Classifier Metrics")
            log.debug(metrics.classification_report(target_test, multinomial_nb_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("Multinomial Naive Bayes Classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, multinomial_nb_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Multinomial Naive Bayes Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Multinomial Naive Bayes Classifier:")
        make_predictions(multinomial_nb_clf)


################################################################################################################
def sgd_classifier():
    """
    Function trains a Stochastic Gradient Descent Classifier.

    Note: "hinge" loss does not support probability estimates.

    :return: none.
    """
    from sklearn.linear_model import SGDClassifier

    sgd_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier(alpha=0.1, average=False, class_weight=None,
                              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=False,
                              l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=5,
                              n_iter=None, n_iter_no_change=5, n_jobs=-1, penalty='l2',
                              power_t=0.5, random_state=None, shuffle=True, tol=1e-1,
                              validation_fraction=0.1, verbose=0, warm_start=False)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(sgd_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Stochastic Gradient Descent: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):

        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        sgd_classifier_clf.fit(tweet_train, target_train)
        sgd_classifier_predictions = sgd_classifier_clf.predict(tweet_test)
        sgd_classifier_predictions_probabilities = sgd_classifier_clf.predict_proba(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations(sgd_classifier_clf, sgd_classifier_predictions,
                                         sgd_classifier_predictions_probabilities,
                                         'Stochastic Gradient Descent Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(sgd_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Stochastic Gradient Descent Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("SGD_classifier Classifier Metrics")
            log.debug(metrics.classification_report(target_test, sgd_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("SGD_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, sgd_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Stochastic Gradient Descent Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Stochastic Gradient Descent Classifier:")
        make_predictions(sgd_classifier_clf)


################################################################################################################
def svm_support_vector_classification():
    """
    Functions trains a Support Vector Machine - Support Vector Classification Classifier.

    :return: none.
    """
    from sklearn import svm

    svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', svm.SVC(C=0.9, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovo', degree=3, gamma='scale', kernel='sigmoid',
                        max_iter=-1, probability=True, random_state=None, shrinking=True,
                        tol=0.01, verbose=False)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(svc_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Support Vector Classification: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        svc_classifier_clf.fit(tweet_train, target_train)
        svc_classifier_predictions = svc_classifier_clf.predict(tweet_test)
        svc_classifier_predictions_probabilities = svc_classifier_clf.predict_proba(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations(svc_classifier_clf, svc_classifier_predictions,
                                         svc_classifier_predictions_probabilities,
                                         'Support Vector Classification Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(svc_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Support Vector Classification Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("SVC_classifier Metrics")
            log.debug(metrics.classification_report(target_test, svc_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("SVC_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, svc_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Support Vector Classification Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Support Vector Classification Classifier:")
        make_predictions(svc_classifier_clf)


################################################################################################################
def svm_linear_support_vector_classification():
    """"
    Function trains a Support Vector Machine - Linear Support Vector Classification Classifier.

    Note: LinearSVC does not have attribute predict_proba.

    :return: none.
    """
    from sklearn import svm

    linear_svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', svm.LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
                              intercept_scaling=1, loss='squared_hinge', max_iter=2000,
                              multi_class='ovr', penalty='l2', random_state=None, tol=0.1,
                              verbose=0)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(linear_svc_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Linear Support Vector Classification: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        linear_svc_classifier_clf.fit(tweet_train, target_train)
        linear_svc_classifier_predictions = linear_svc_classifier_clf.predict(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations_linearsvc(linear_svc_classifier_clf, linear_svc_classifier_predictions,
                                                   'Linear Support Vector Classification Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(linear_svc_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Linear Support Vector Classification Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("LinearSVC_classifier Metrics")
            log.debug(metrics.classification_report(target_test, linear_svc_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("LinearSVC_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, linear_svc_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Linear Support Vector Classification Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Linear Support Vector Classification Classifier:")
        make_predictions(linear_svc_classifier_clf)


################################################################################################################
def nearest_kneighbor_classifier():
    """
    Function trains a Nearest Neighbor - KNeighbor Classifier.

    :return: none.
    """
    from sklearn.neighbors import KNeighborsClassifier

    k_neighbor_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', KNeighborsClassifier(n_neighbors=30, algorithm='auto', leaf_size=10, metric='euclidean', p=1,
                                     weights='uniform', n_jobs=-1)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(k_neighbor_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("KNeighbor: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        k_neighbor_classifier_clf.fit(tweet_train, target_train)
        k_neighbor_classifier_predictions = k_neighbor_classifier_clf.predict(tweet_test)
        k_neighbor_classifier_predictions_probabilities = k_neighbor_classifier_clf.predict_proba(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations_kneighbor(k_neighbor_classifier_clf, k_neighbor_classifier_predictions,
                                                   k_neighbor_classifier_predictions_probabilities,
                                                   'KNeighbor Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(k_neighbor_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using KNeighbor Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("KNeighbor_classifier Metrics")
            log.debug(metrics.classification_report(target_test, k_neighbor_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("KNeighbor_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, k_neighbor_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("KNeighbor Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using KNeighbor Classifier:")
        make_predictions(k_neighbor_classifier_clf)


################################################################################################################
def decision_tree_classifier():
    """
    Functions trains a Decision Tree Classifier.

    :return: none.
    """
    from sklearn import tree

    decision_tree_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', tree.DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None,
                                            max_leaf_nodes=3, min_impurity_decrease=1e-5, min_samples_leaf=1,
                                            min_samples_split=2, min_weight_fraction_leaf=0)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(decision_tree_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Decision Tree: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        decision_tree_classifier_clf.fit(tweet_train, target_train)
        decision_tree_classifier_predictions = decision_tree_classifier_clf.predict(tweet_test)
        decision_tree_classifier_predictions_probabilities = decision_tree_classifier_clf.predict_proba(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations(decision_tree_classifier_clf, decision_tree_classifier_predictions,
                                         decision_tree_classifier_predictions_probabilities,
                                         'Decision Tree Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(decision_tree_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Decision Tree Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("DecisionTree_classifier Metrics")
            log.debug(metrics.classification_report(target_test, decision_tree_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("DecisionTree_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, decision_tree_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Decision Tree Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Decision Tree Classifier:")
        make_predictions(decision_tree_classifier_clf)


################################################################################################################
def multi_layer_perceptron_classifier():
    """
    Function trains a Multi Layer Perceptron Neural Network Classifier.

    :return: none.
    """
    from sklearn.neural_network import MLPClassifier

    mlp_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MLPClassifier(activation='identity', alpha=1e-1, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant', learning_rate_init=1e-1,
                              max_iter=1000, momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='lbfgs', tol=0.1,
                              validation_fraction=0.1, verbose=False, warm_start=False)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(mlp_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Multi Layer Perceptron Neural Network: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        # scaler.fit(tweet_train)
        # tweet_train_scaled = scaler.transform(tweet_train)
        # tweet_test_scaled = scaler.transform(tweet_test)

        mlp_classifier_clf.fit(tweet_train, target_train)
        mlp_classifier_predictions = mlp_classifier_clf.predict(tweet_test)
        mlp_classifier_predictions_probabilities = mlp_classifier_clf.predict_proba(tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations(mlp_classifier_clf, mlp_classifier_predictions,
                                         mlp_classifier_predictions_probabilities,
                                         'Multi Layer Perceptron Neural Network Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(mlp_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Decision Tree Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("MLP_classifier Metrics")
            log.debug(metrics.classification_report(target_test, mlp_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("MLP_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, mlp_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Multi Layer Perceptron Neural Network Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Multi Layer Perceptron Neural Network Classifier:")
        make_predictions(mlp_classifier_clf)


################################################################################################################
def logistic_regression_classifier():
    """
    Function trains a Logistic Regression Classifier.

    :return: none.
    """
    from sklearn.linear_model import LogisticRegression

    logistic_regression_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', LogisticRegression(C=1.0, class_weight=None, fit_intercept=False, max_iter=2000,
                                   multi_class='ovr', penalty='l2', solver='sag', tol=1e-1)),
    ])

    from sklearn.model_selection import cross_val_score

    # Compute cross-validation metrics.
    create_training_and_test_set()
    cross_val_scores = cross_val_score(logistic_regression_classifier_clf, tweet_train, target_train, cv=5)
    # Display results.
    log.debug("Logistic Regression: ")
    log.debug("Cross validated metrics (5-fold cross validation):")
    log.debug("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    log.debug("\n")

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    for index in range(0, iterations):

        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        logistic_regression_classifier_clf.fit(tweet_train, target_train)
        logistic_regression_classifier_predictions = logistic_regression_classifier_clf.predict(tweet_test)
        logistic_regression_classifier_predictions_probabilities = logistic_regression_classifier_clf.predict_proba(
            tweet_test)

        if enable_visualizations:
            # Visualize the results of training our model.
            create_metric_visualizations(logistic_regression_classifier_clf, logistic_regression_classifier_predictions,
                                         logistic_regression_classifier_predictions_probabilities,
                                         'Logistic Regression Classifier')

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(logistic_regression_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Logistic Regression Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("LogisticRegression_classifier Metrics")
            log.debug(metrics.classification_report(target_test, logistic_regression_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("LogisticRegression_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, logistic_regression_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Logistic Regression Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    if enable_predictions:
        log.debug("Prediction statistics using Logistic Regression Classifier:")
        make_predictions(logistic_regression_classifier_clf)


################################################################################################################
def keras_deep_neural_network():
    """
    Function implements a Keras Deep Neural Network Model.
    FIXME - non-functional at the moment.
    :return: none.
    """

    from keras.models import Sequential
    from keras import layers

    from keras.models import Sequential
    from keras.layers import Dense, Embedding, SimpleRNN

    create_training_and_test_set()

    model = Sequential()
    model.add(Embedding(1000, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(tweet_train, target_train,  # TODO - need to convert to numerical data first.
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


################################################################################################################

############################################################################################
"""
Main function.  Execute the program.
"""
import time

if __name__ == '__main__':

    # For debug purposes.
    # my_set = create_prediction_set()

    start_time = time.time()

    # Call non-pipelined multinomial Naive Bayes Classifier training function.
    # scikit_learn_multinomialnb_classifier_non_pipeline()

    ################################################
    """
    This section calls Scikit-Learn classifier functions for model training and prediction.
    """
    # multinomial_naive_bayes_classifier()
    # sgd_classifier()
    # svm_support_vector_classification()
    # svm_linear_support_vector_classification()
    # nearest_kneighbor_classifier()
    # decision_tree_classifier()
    # multi_layer_perceptron_classifier()
    # logistic_regression_classifier()

    ################################################

    # Keras RNN.
    # keras_deep_neural_network()

    end_time = time.time()

    if debug_pipeline:
        log.debug("The time taken to train the classifier(s), make predictions, and visualize the results is:")
        total_time = end_time - start_time
        log.debug(str(total_time))
        log.debug("\n")

############################################################################################
