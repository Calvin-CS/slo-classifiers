"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 8-1-19

Gensim: HDP - Hierarchical Dirichlet Process

###########################################################
Notes:

###########################################################
Resources Used:

https://radimrehurek.com/gensim/models/hdpmodel.html
https://www.machinelearningplus.com/nlp/gensim-tutorial/
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

https://stackoverflow.com/questions/509211/understanding-slice-notation

"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import time
import warnings

import pandas as pd
import seaborn as sns
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer

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

def hierarchical_dirichlet_process_topic_extraction():
    """
    Function performs topic extraction on Tweets using the Gensim HDP model.

    :return: None.
    """
    from gensim.models import HdpModel

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(slo_feature_series)
    tf_feature_names = tf_vectorizer.get_feature_names()

    log.info("\n.fit_transform - Learn the vocabulary dictionary and return term-document matrix.")
    log.info(f"{tf}\n")
    log.info("\n.get_feature_names - Array mapping from feature integer indices to feature name")
    log.info(f"{tf_feature_names}\n")

    # Train the HDP model.
    hdp = HdpModel(corpus, dictionary)
    time.sleep(3)

    # # For use as wrapper with Scikit-Learn API.
    # model = HdpTransformer(id2word=dictionary)
    # distribution = model.fit_transform(corpus)

    # Display the top words for each topic.
    topic_info = hdp.print_topics(num_topics=20, num_words=10)

    for topic in topic_info:
        print(topic)


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
    hierarchical_dirichlet_process_topic_extraction()

    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
          f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")

############################################################################################
