"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-8-19

Gensim: ATM - Author-Topic Model.

###########################################################
Notes:

TODO - remove stop words before training.

###########################################################
Resources Used:

https://www.machinelearningplus.com/nlp/gensim-tutorial/
https://radimrehurek.com/gensim/models/atmodel.html
https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb

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

def author_topic_model_topic_extraction():
    """
    Function performs topic extraction on Tweets using the Gensim Author-Topic model.

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
    # author_topic_model_topic_extraction()

    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
          f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")

############################################################################################
