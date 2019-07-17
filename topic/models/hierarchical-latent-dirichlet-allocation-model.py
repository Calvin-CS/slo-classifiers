"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-8-19

HlDA - Hierarchical Latent Dirichlet Allocation

###########################################################
Notes:

TODO - remove stop words before training.
FIXME - not functional.

###########################################################
Resources Used:

https://github.com/jreades/hlda (updated version for Python 3)

"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import warnings
import time
import pandas as pd
import spacy
import numpy as np
from wordcloud import WordCloud
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
log.disable(level=log.DEBUG)

################################################################################################################
################################################################################################################

# # Import the dataset (relative path).
# tweet_dataset_processed = \
#     pd.read_csv("twitter-dataset-7-10-19-lda-ready-tweet-text-with-hashtags-excluded-created-7-17-19.csv", sep=",")

# Import the dataset (absolute path).
tweet_dataset_processed = \
    pd.read_csv("D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
                "twitter-dataset-7-10-19-lda-ready-tweet-text-with-hashtags-excluded-created-7-17-19.csv", sep=",")

# # Import the dataset (test/debug).
# tweet_dataset_processed = \
#     pd.read_csv("twitter-dataset-7-10-19-lda-ready-tweet-text-test.csv", sep=",")

# # Import the dataset (test/debug).
# tweet_dataset_processed = \
#     pd.read_csv("D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#                 "twitter-dataset-7-10-19-lda-ready-tweet-text-test.csv", sep=",")

# Reindex and shuffle the data randomly.
tweet_dataset_processed = tweet_dataset_processed.reindex(
    pd.np.random.permutation(tweet_dataset_processed.index))

# Generate a Pandas dataframe.
tweet_text_dataframe = pd.DataFrame(tweet_dataset_processed)

# # Print shape and column names.
# log.info(f"\nThe shape of the Tweet text dataframe:")
# log.info(f"{tweet_text_dataframe.shape}\n")
# log.info(f"\nThe columns of the Tweet text dataframe:")
# log.info(f"{tweet_text_dataframe.columns}\n")

# Print shape and column names.
log.info("\nThe shape of the Tweet text dataframe:")
log.info(tweet_text_dataframe.shape)
log.info("\nThe columns of the Tweet text dataframe:")
log.info(tweet_text_dataframe.columns)

# Drop any NaN or empty Tweet rows in dataframe (or else CountVectorizer will blow up).
tweet_text_dataframe = tweet_text_dataframe.dropna()

# # Print shape and column names.
# log.info(f"\nThe shape of the Tweet text dataframe with NaN (empty) rows dropped:")
# log.info(f"{tweet_text_dataframe.shape}\n")
# log.info(f"\nThe columns of the Tweet text dataframe with NaN (empty) rows dropped:")
# log.info(f"{tweet_text_dataframe.columns}\n")

# Print shape and column names.
log.info("\nThe shape of the Tweet text dataframe with NaN (empty) rows dropped:")
log.info(tweet_text_dataframe.shape)
log.info("\nThe columns of the Tweet text dataframe with NaN (empty) rows dropped:")
log.info(tweet_text_dataframe.columns)

# Reindex everything.
tweet_text_dataframe.index = pd.RangeIndex(len(tweet_text_dataframe.index))

# Assign column names.
tweet_text_dataframe_column_names = ['text_derived', 'text_derived_preprocessed', 'text_derived_postprocessed']

# Rename column in dataframe.
tweet_text_dataframe.columns = tweet_text_dataframe_column_names

# Create input feature.
selected_features = tweet_text_dataframe[['text_derived_postprocessed']]
processed_features = selected_features.copy()

# # Check what we are using as inputs.
# log.info(f"\nA sample Tweet in our input feature:")
# log.info(f"{processed_features['text_derived_postprocessed'][0]}\n")

# Check what we are using as inputs.
log.info("\nA sample Tweet in our input feature:")
log.info(processed_features['text_derived_postprocessed'][0])

# Create feature set.
slo_feature_series = processed_features['text_derived_postprocessed']
slo_feature_series = pd.Series(slo_feature_series)
slo_feature_list = slo_feature_series.tolist()

#############################################################

corpus = []
dictionary = set()
nlp = spacy.load('en')
nlp.remove_pipe("parser")
nlp.remove_pipe("tagger")
nlp.remove_pipe("ner")

# Create the corpus of documents and dictionary of words (vocabulary)
for tweet in slo_feature_list:
    # Tokenize each Tweet (document) and add to List of documents in the corpus.
    corpus.append(tweet.split())
    # Tokenize each Tweet (document) and add individual words to the dictionary of words (vocabulary).
    dictionary.update(tweet.split())

# Attach indices to each word to represent their position in the dictionary of words (vocabulary).
dictionary = sorted(list(dictionary))
vocab_index = {}
for i, w in enumerate(dictionary):
    vocab_index[w] = i

# print(f"\nThe number of documents: {len(slo_feature_list)}")
# print(f"\nThe number of words in the dictionary: {len(dictionary)}")
# print(f"Sample of the words in the dictionary:\n {dictionary[0:100]}")
# print(f"\nThe number of documents in the corpus: {len(corpus)}")
# print(f"Sample of the documents in the corpus:\n {corpus}")

print("\nThe number of documents: " + str(len(slo_feature_list)))
print("\nThe number of words in the dictionary: " + str(len(dictionary)))
print("Sample of the words in the dictionary:\n " + str(dictionary[0:100]))
print("\nThe number of documents in the corpus: " + str(len(corpus)))
print("Sample of the documents in the corpus:\n " + str(corpus))

# Visualize the dictionary of words.
wordcloud = WordCloud(background_color='white').generate(' '.join(slo_feature_list))
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

print("\nLength of the dictionary, corpus, document 0 in corpus, document 1 in corpus (in that order)")
print(len(dictionary), len(corpus), len(corpus[0]), len(corpus[1]))

"""
Modify the corpus of documents to store the index value of each word from the dictionary (vocabulary)
rather than the words themselves.
"""
new_corpus = []
for document in corpus:
    new_document = []
    for word in document:
        word_index = vocab_index[word]
        new_document.append(word_index)
    new_corpus.append(new_document)

print("\nLength of the dictionary and corpus (as word dictionary index values (in that order))")
print(len(dictionary), len(new_corpus))

print("\nDocument 0 in the corpus as tokenized words:")
print(corpus[0][0:10])
print("Document 0 in the corpus as tokenized word index values from the dictionary:")
print(new_corpus[0][0:10])

print("\nDocument 1 in the corpus as tokenized words:")
print(corpus[1][0:10])
print("Document 1 in the corpus as tokenized word index values from the dictionary:")
print(new_corpus[1][0:10])

print("\nDocument 2 in the corpus as tokenized words:")
print(corpus[2][0:10])
print("Document 2 in the corpus as tokenized word index values from the dictionary:")
print(new_corpus[2][0:10])


################################################################################################################

def hierarchical_latent_dirichlet_allocation_topic_extraction():
    """
    Function performs topic extraction on Tweets using the Gensim HDP model.

    :return: None.
    """
    from hlda.sampler import HierarchicalLDA

    # Set parameters.
    n_samples = 500  # no of iterations for the sampler
    alpha = 10.0  # smoothing over level distributions
    gamma = 1.0  # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
    eta = 0.1  # smoothing over topic-word distributions
    num_levels = 3  # the number of levels in the tree
    display_topics = 50  # the number of iterations between printing a brief summary of the topics so far
    n_words = 5  # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights

    # Train the model.
    hlda = HierarchicalLDA(new_corpus, dictionary, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
    hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)


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
    hierarchical_latent_dirichlet_allocation_topic_extraction()

    ################################################
    my_end_time = time.time()

    time_elapsed_in_seconds = (my_end_time - my_start_time)
    time_elapsed_in_minutes = (my_end_time - my_start_time) / 60.0
    time_elapsed_in_hours = (my_end_time - my_start_time) / 60.0 / 60.0
    # print(f"Time taken to process dataset: {time_elapsed_in_seconds} seconds, "
    #       f"{time_elapsed_in_minutes} minutes, {time_elapsed_in_hours} hours.")
    print("\n\nTime taken to process dataset: " + str(time_elapsed_in_seconds) + " seconds, " +
          str(time_elapsed_in_minutes) + " minutes, " + str(time_elapsed_in_hours) + " hours.\n")

############################################################################################
