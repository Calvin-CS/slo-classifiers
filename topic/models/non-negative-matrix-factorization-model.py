"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 8-1-19

Scikit-Learn: NMF - Non-negative Matrix Factorization

###########################################################
Notes:

Based on linear algebra, not a statistical model.

###########################################################
Resources Used:

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import time
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Import custom utility functions.
import topic_extraction_utility_functions as topic_util

#############################################################

# Pandas options.
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = 1000
# Pandas float precision display.
pd.set_option('precision', 12)
# Don't output these types of warnings to terminal.
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# Matplotlib log settings.
mylog = log.getLogger("matplotlib")
mylog.setLevel(log.INFO)

"""
Turn debug log statements for various sections of code on/off.
(adjust log level as necessary)
"""
log.basicConfig(level=log.INFO)
log.disable(level=log.DEBUG)

################################################################################################################
################################################################################################################

# Import the dataset (absolute path).
tweet_dataset_processed = \
    pd.read_csv("D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
                "twitter-dataset-7-10-19-topic-extraction-ready-tweet-text-with-hashtags-excluded"
                "-created-7-29-19-tokenized.csv", sep=",")

# # Import the dataset (test/debug).
# tweet_dataset_processed = \
#     pd.read_csv("D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#                 "twitter-dataset-7-10-19-topic-extraction-ready-tweet-text-with-hashtags-excluded"
#                 "-created-7-30-19-test.csv", sep=",")

# Reindex and shuffle the data randomly.
tweet_dataset_processed = tweet_dataset_processed.reindex(
    pd.np.random.permutation(tweet_dataset_processed.index))

# Generate a Pandas dataframe.
tweet_text_dataframe = pd.DataFrame(tweet_dataset_processed)

# Print shape and column names.
log.info(f"\nThe shape of the Tweet text dataframe:")
log.info(f"{tweet_text_dataframe.shape}\n")
log.info(f"\nThe columns of the Tweet text dataframe:")
log.info(f"{tweet_text_dataframe.columns}\n")

# Drop any NaN or empty Tweet rows in dataframe (or else CountVectorizer will blow up).
tweet_text_dataframe = tweet_text_dataframe.dropna()

# Print shape and column names.
log.info(f"\nThe shape of the Tweet text dataframe with NaN (empty) rows dropped:")
log.info(f"{tweet_text_dataframe.shape}\n")
log.info(f"\nThe columns of the Tweet text dataframe with NaN (empty) rows dropped:")
log.info(f"{tweet_text_dataframe.columns}\n")

# Reindex everything.
tweet_text_dataframe.index = pd.RangeIndex(len(tweet_text_dataframe.index))

# Assign column names.
tweet_text_dataframe_column_names = ['text_derived', 'text_derived_preprocessed', 'text_derived_postprocessed']

# Rename column in dataframe.
tweet_text_dataframe.columns = tweet_text_dataframe_column_names

# Create input feature.
selected_features = tweet_text_dataframe[['text_derived_postprocessed']]
processed_features = selected_features.copy()

# Check what we are using as inputs.
log.info(f"\nA sample Tweet in our input feature:")
log.info(f"{processed_features['text_derived_postprocessed'][0]}\n")

# Create feature set.
slo_feature_series = processed_features['text_derived_postprocessed']
slo_feature_series = pd.Series(slo_feature_series)
slo_feature_list = slo_feature_series.tolist()


################################################################################################################

def non_negative_matrix_factorization_topic_extraction():
    """
    Function performs topic extraction on Tweets using Scikit-Learn NMF model.

    :return: None.
    """
    from sklearn.decomposition import NMF

    # Use tf-idf features for NMF.
    print("\nExtracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(slo_feature_series)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Run NMF using Frobenius norm.
    nmf_frobenius = NMF(n_components=20, random_state=1,
                        alpha=.1, l1_ratio=.5).fit(tfidf)

    # Run NMF using generalized Kullback-Leibler divergence.
    nmf_kullback_leibler = NMF(n_components=20, random_state=1,
                               beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                               l1_ratio=.5).fit(tfidf)

    time.sleep(3)

    # Display the top words for each topic.
    print("\nTopics using NMF Frobenius norm:")
    topic_util.display_topics(nmf_frobenius, tfidf_feature_names, 10)

    # Display the top words for each topic.
    print("\nTopics using generalized Kullback-Leibler divergence:")
    topic_util.display_topics(nmf_kullback_leibler, tfidf_feature_names, 10)


################################################################################################################

############################################################################################

"""
Main function.  Execute the program.
"""
if __name__ == '__main__':
    my_start_time = time.time()
    ################################################
    """
    Perform exhaustive grid search.
    """
    # FIXME - non functional unless we find a way to disable cross-validation "cv" parameter in GridSearchCV Class.
    # What parameters do we search for?
    lda_search_parameters = {
        'vect__strip_accents': [None],
        'vect__lowercase': [True],
        'vect__stop_words': ['english'],
        # 'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'vect__analyzer': ['word'],
        'vect__min_df': [2],
        'vect__max_df': [0.95],
        'vect__max_features': [1000],
        'clf__n_components': [5, 10, 20],
        'clf__init': ['random', 'nndsvd', 'nndsvda', 'nndsvdar'],
        'clf__solver': ['cd', 'mu'],
        'clf__beta_loss': ['frobenius', 'kullback-leibler', 'itakura-saito'],
        'clf__tol': [1e-2, 1e-4, 1e-6],
        'clf__max_iter': [100, 200, 300],
        'clf__alpha': [0],
        'clf__l1_ratio': [0],
        'clf__verbose': [False],
        'clf__shuffle': [False],
        'clf__random_state': [None],
    }
    # topic_util.non_negative_matrix_factorization_grid_search(slo_feature_series, lda_search_parameters)
    """
    Perform exhaustive grid search on data subset.
    """
    # data_subset = topic_util.dataframe_subset(tweet_dataset_processed, 50)
    # topic_util.non_negative_matrix_factorization_grid_search(data_subset, lda_search_parameters)

    """
    Perform the topic extraction.
    """
    # non_negative_matrix_factorization_topic_extraction()

    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
          f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")

############################################################################################
