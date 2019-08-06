"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 8-1-19

Scikit-Learn: LDA - Latent Dirichlet Allocation

###########################################################
Notes:

LDA can only use raw term counts (CANNOT use tfidf transformer)

###########################################################
Resources Used:

https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
https://pypi.org/project/lda/

TODO - setup Singularity container and perform exhaustive grid search or randomized grid search for LDA hyperparameters.
"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Import custom utility functions.
import topic_extraction_utility_functions as lda_util

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

def latent_dirichlet_allocation_topic_extraction():
    """
    Function performs topic extraction on Tweets using Scikit-Learn LDA model.

    :return: None.
    """
    from sklearn.decomposition import LatentDirichletAllocation

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(slo_feature_series)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Run LDA.
    lda = LatentDirichletAllocation(n_components=20, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    time.sleep(3)

    # Display the top words for each topic.
    lda_util.display_topics(lda, tf_feature_names, 10)

    import pyLDAvis
    from pyLDAvis import sklearn
    # pyLDAvis.enable_notebook()
    visualization = sklearn.prepare(lda_model=lda, vectorizer=tf_vectorizer, dtm=tf)
    pyLDAvis.save_html(visualization, 'lda_visualization-no-company-words.html')


################################################################################################################

def latent_dirichlet_allocation_collapsed_gibbs_sampling():
    """
    Functions performs LDA topic extraction using collapsed Gibbs Sampling.

    https://pypi.org/project/lda/

    :return: None.
    """
    import lda

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(slo_feature_series)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Train and fit the LDA model.
    model = lda.LDA(n_topics=12, n_iter=1000, random_state=1)
    model.fit(tf)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 10
    time.sleep(3)

    # Display the topics and the top words associated with.
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))


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
        'clf__doc_topic_prior': [None],
        'clf__topic_word_prior': [None],
        'clf__learning_method': ['online'],
        'clf__learning_decay': [0.5, 0.7, 0.9],
        'clf__learning_offset': [5, 10, 15],
        'clf__max_iter': [1, 5, 10],
        'clf__batch_size': [64, 128, 256],
        'clf__evaluate_every': [0],
        # 'clf__total_samples': [1e4, 1e6, 1e8],
        # 'clf__perp_tol': [1e-1, 1e-2, 1e-3],
        'clf__mean_change_tol': [1e-1, 1e-3, 1e-5],
        'clf__max_doc_update_iter': [50, 100, 150],
        'clf__n_jobs': [-1],
        'clf__verbose': [0],
        'clf__random_state': [None],
    }
    # lda_util.latent_dirichlet_allocation_grid_search(slo_feature_series, lda_search_parameters)
    """
    Perform exhaustive grid search on data subset.
    """
    # data_subset = lda_util.dataframe_subset(tweet_dataset_processed, 50)
    # lda_util.latent_dirichlet_allocation_grid_search(data_subset, lda_search_parameters)

    """
    Perform the topic extraction.
    """
    latent_dirichlet_allocation_topic_extraction()
    """
    Perform the topic extraction using collapsed Gibbs Sampling.
    """
    # latent_dirichlet_allocation_collapsed_gibbs_sampling()
    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
          f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")

############################################################################################
