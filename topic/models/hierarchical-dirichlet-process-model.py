"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-8-19

Gensim: HDP - Hierarchical Dirichlet Process

###########################################################
Notes:

TODO - remove stop words before training.

###########################################################
Resources Used:

https://radimrehurek.com/gensim/
https://www.machinelearningplus.com/nlp/gensim-tutorial/
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

https://stackoverflow.com/questions/509211/understanding-slice-notation

Initial Results (default hyper parameters, with stop words included):

[(0, '0.043*the + 0.037*to + 0.018*a + 0.018*of + 0.017*in + 0.016*is + 0.015*for + 0.015*and + 0.012*coal + 0.011*mine'),
 (1, '0.038*the + 0.030*to + 0.020*of + 0.017*in + 0.017*a + 0.014*for + 0.013*and + 0.012*is + 0.010*coal + 0.010*on'),
 (2, '0.034*the + 0.030*to + 0.016*of + 0.015*in + 0.014*a + 0.013*for + 0.012*and + 0.012*is + 0.010*coal + 0.009*on'),
 (3, '0.029*the + 0.029*��� + 0.026*tax + 0.024*to + 0.015*on + 0.012*of + 0.012*in + 0.012*and + 0.011*a + 0.011*for'),
 (4, '0.032*the + 0.028*to + 0.015*of + 0.014*a + 0.014*in + 0.013*for + 0.011*and + 0.011*is + 0.009*on + 0.009*coal'),
 (5, '0.029*the + 0.026*to + 0.015*in + 0.012*of + 0.012*and + 0.012*a + 0.011*for + 0.010*on + 0.009*is + 0.008*coal'),
 (6, '0.033*the + 0.029*to + 0.015*of + 0.015*a + 0.014*in + 0.013*for + 0.012*and + 0.012*is + 0.010*coal + 0.009*on'),
 (7, '0.033*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.012*is + 0.010*coal + 0.009*on'),
 (8, '0.032*the + 0.029*to + 0.014*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.011*is + 0.010*coal + 0.010*on'),
 (9, '0.033*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.011*is + 0.010*coal + 0.009*on'),
 (10, '0.033*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.012*is + 0.010*coal + 0.009*on'),
 (11, '0.032*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.011*and + 0.011*is + 0.009*coal + 0.009*on'),
 (12, '0.032*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.011*is + 0.010*coal + 0.009*on'),
 (13, '0.033*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.011*and + 0.011*is + 0.010*coal + 0.009*on'),
 (14, '0.032*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.012*for + 0.012*and + 0.011*is + 0.010*coal + 0.009*on'),
 (15, '0.033*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.011*is + 0.010*coal + 0.009*on'),
 (16, '0.032*the + 0.029*to + 0.014*of + 0.014*in + 0.014*a + 0.013*for + 0.012*and + 0.011*is + 0.010*coal + 0.009*on'),
 (17, '0.032*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.011*is + 0.011*and + 0.010*coal + 0.009*on'),
 (18, '0.032*the + 0.029*to + 0.015*of + 0.014*in + 0.014*a + 0.013*for + 0.011*and + 0.011*is + 0.010*coal + 0.009*on'),
 (19, '0.032*the + 0.029*to + 0.014*of + 0.014*in + 0.013*a + 0.013*for + 0.011*is + 0.011*and + 0.010*coal + 0.009*on')]

 Time taken to process dataset: 2175.6769523620605 seconds, 36.26128253936768 minutes, 0.6043547089894613 hours
"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import warnings
import time
import pandas as pd
import numpy as np
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import seaborn as sns

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
# Seaborn setting.
sns.set()
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


################################################################################################################
################################################################################################################

# Import the dataset.
tweet_dataset_processed = \
    pd.read_csv("D:/Dropbox/summer-research-2019/datasets/dataset_20100101-20180510_tok_LDA_PROCESSED.csv", sep=",")

# Reindex and shuffle the data randomly.
tweet_text_dataframe = tweet_dataset_processed.reindex(
    pd.np.random.permutation(tweet_dataset_processed.index))

# Generate a Pandas dataframe.
tweet_dataframe_processed = pd.DataFrame(tweet_dataset_processed)

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
tweet_text_dataframe_column_names = ['Tweet']

# Rename column in dataframe.
tweet_text_dataframe.columns = tweet_text_dataframe_column_names

# Create input feature.
selected_features = tweet_text_dataframe[['Tweet']]
processed_features = selected_features.copy()

# Check what we are using as inputs.
log.info(f"\nA sample Tweet in our input feature:")
log.info(f"{processed_features['Tweet'][0]}\n")

# Create feature set.
slo_feature_series = processed_features['Tweet']
slo_feature_series = pd.Series(slo_feature_series)
slo_feature_list = slo_feature_series.tolist()

# Convert feature list of sentences to comma-separated dictionary of words.
words = [[text for text in tweet.split()] for tweet in slo_feature_list]
log.info(f"\nDictionary of individual words:")
log.info(f"{words[0]}\n")

# Create the Gensim dictionary of words.
dictionary = corpora.Dictionary(words)
log.info(f"\nGensim dictionary of tokenized words.")
log.info(f"{dictionary}\n")

# Create the Gensim corpus of document term frequencies.
corpus = [dictionary.doc2bow(word, allow_update=True) for word in words]
log.info(f"\nGensim corpus of document-term frequencies.")
log.info(f"{corpus[0:10]}\n")


################################################################################################################

def hierarchical_dirichlet_process_topic_extraction():
    """
    Function performs topic extraction on Tweets using the Gensim HDP model.

    :return: None.
    """
    from gensim.test.utils import common_corpus, common_dictionary
    from gensim.models import HdpModel
    from gensim.sklearn_api import HdpTransformer

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(slo_feature_series)
    tf_feature_names = tf_vectorizer.get_feature_names()

    log.info("\n.fit_transform - Learn the vocabulary dictionary and return term-document matrix.")
    log.info(f"{tf}\n")
    log.info("\n.get_feature_names - Array mapping from feature integer indices to feature name")
    log.info(f"{tf_feature_names}\n")

    # Sample dictionary and corpus.
    log.info(f"\nExample dictionary format for Gensim:")
    log.info(f"{common_dictionary}\n")
    log.info(f"\nExample corpus format for Gensim:")
    log.info(f"{common_corpus}\n")

    # Train the HDP model.
    hdp = HdpModel(corpus, dictionary)

    # # For use with Scikit-Learn API.
    # model = HdpTransformer(id2word=dictionary)
    # distribution = model.fit_transform(corpus)

    # Display the top words for each topic.
    topic_info = hdp.print_topics(num_topics=20, num_words=10)
    print(topic_info)


############################################################################################

"""
Main function.  Execute the program.
"""
if __name__ == '__main__':
    my_start_time = time.time()
    ################################################

    """
    Perform the Twitter dataset preprocessing.
    """
    # lda_util.tweet_dataset_preprocessor(
    #     "D:/Dropbox/summer-research-2019/datasets/dataset_20100101-20180510_tok_PROCESSED.csv",
    #     "D:/Dropbox/summer-research-2019/datasets/dataset_20100101-20180510_tok_LDA_PROCESSED2.csv", "tweet_t")

    """
    Perform the topic extraction.
    """
    hierarchical_dirichlet_process_topic_extraction()

    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
          f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")

############################################################################################
