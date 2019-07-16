"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-8-19

Gensim: ATM - Author-Topic Model.

###########################################################
Notes:

FIXME - IndexError: list index out of range (referring to author2doc mappings, I think)

Possible fixes:

Create Dictionary mapping unique author screen-names to the row index value of the dataset who they are the author of.

Then, the author2doc data structure should map each author to a List of those row index values instead of Tweet ID's.

Then, create a Gensim corpus of documents that associates each Tweet with their respective row index values.

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
import topic_extraction_utility_functions as topic_util
import slo_twitter_data_analysis_utility_functions as tweet_util_v2

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

# Import untokenized CSV dataset.
tweet_dataset_untokenized = tweet_util_v2.import_dataset(
    "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
    "twitter-dataset-7-10-19-test-subset-100-examples.csv",
    "csv", False)

# Create author-document mappings as a dictionary of key: author, values: tweet ID's
author2doc = topic_util.topic_author_model(tweet_dataset_untokenized, False)

# Import tokenized CSV dataset.
tweet_dataset_tokenized = tweet_util_v2.import_dataset(
    "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
    "twitter-dataset-7-10-19-lda-ready-tweet-text-test.csv",
    "csv", False)

# Reindex and shuffle the data randomly.
tweet_text_dataframe = tweet_dataset_tokenized.reindex(
    pd.np.random.permutation(tweet_dataset_tokenized.index))

# Generate a Pandas dataframe.
tweet_dataframe_processed = pd.DataFrame(tweet_dataset_tokenized)

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

# # Assign column names.
# tweet_text_dataframe_column_names = ['Tweet']
#
# # Rename column in dataframe.
# tweet_text_dataframe.columns = tweet_text_dataframe_column_names

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

# Convert feature list of sentences to comma-separated dictionary of words.
words = [[text for text in tweet.split()] for tweet in slo_feature_list]
log.info(f"\nDictionary of individual words:")
log.info(f"{words[0]}\n")

# Create the Gensim dictionary of words.
dictionary = corpora.Dictionary(words)
log.info(f"\nGensim dictionary of tokenized words.")
log.info(f"{dictionary}\n")
log.info(f"\nGensim dictionary of tokenized words with index ID's.")
log.info(f"{dictionary.token2id}\n")

# Create the Gensim corpus of document term frequencies.
corpus = [dictionary.doc2bow(word, allow_update=True) for word in words]
log.info(f"# of documents in corpus: {len(corpus)}\n")
log.info(f"\nSample of Gensim corpus of document-term frequencies.")
log.info(f"{corpus[0:10]}\n")


################################################################################################################

def author_topic_model_topic_extraction():
    """
    Function performs topic extraction on Tweets using the Gensim Author-Topic model.

    :return: None.
    """
    from gensim.models import AuthorTopicModel

    model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary,
                             author2doc=author2doc, chunksize=2000, passes=1, eval_every=0,
                             iterations=1, random_state=1)

    topic_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for topic in model.show_topics(num_topics=10):
        print(f"'Label: ' + {topic_labels[topic[0]]}")
        wordz = ''
        for word, prob in model.show_topic(topic[0]):
            wordz += word + ' '
        print(f"'Words: ' + {wordz}")


############################################################################################

"""
Main function.  Execute the program.
"""
if __name__ == '__main__':
    my_start_time = time.time()
    ################################################
    """
    Perform the topic extraction.
    """
    author_topic_model_topic_extraction()

    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
          f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")

############################################################################################
