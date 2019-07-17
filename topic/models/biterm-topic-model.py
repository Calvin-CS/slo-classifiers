"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-8-19

Biterm Topic Model

###########################################################
Notes:

TODO - remove stop words before training.

###########################################################
Resources Used:


###########################################################
Sample Results:

 Topic coherence ..
Topic 0 | Coherence=-110.16 | Top words= coal project reef need people energy coming say jobs barrier
Topic 1 | Coherence=-102.36 | Top words= right does business coal thing fight time climate govt lets
Topic 2 | Coherence=-110.99 | Top words= coal project gas seam money news wont investment wants bad
Topic 3 | Coherence=-107.85 | Top words= adanis basin loan jobs islands cou cayman galilee direct profit
Topic 4 | Coherence=-102.73 | Top words= coal aust time finance public pls stop tell breaking trade
Topic 5 | Coherence=-126.78 | Top words= coal jobs climate public power cut project lets poor cou
Topic 6 | Coherence=-114.50 | Top words= coal iron foescue gas ore narrabri project repo oil going
Topic 7 | Coherence=-95.52 | Top words= billion council wont ppl want dollars gov galilee issue australian
Topic 8 | Coherence=-100.79 | Top words= know company pay did people qlders rates ceo failing voting
Topic 9 | Coherence=-85.15 | Top words= barnaby right joyce water rail action sho term suppo gas
Topic 10 | Coherence=-124.35 | Top words= coal water loan adani news government goes dont suppo qlds
Topic 11 | Coherence=-80.77 | Top words= power india tax prices record hit indian solar australian coal
Topic 12 | Coherence=-86.96 | Top words= pay council townsville queensland airstrip loan national coal rail working
Topic 13 | Coherence=-106.44 | Top words= tax need doesnt subsidy loan slocashn australian return dont cash
Topic 14 | Coherence=-78.61 | Top words= labor election shoen want project coal stop win suppo halt
Topic 15 | Coherence=-123.08 | Top words= coal jobs new galilee india goes basin sydney need thousands
Topic 16 | Coherence=-110.03 | Top words= reef want barrier coal new dont save turnbull funding kill
Topic 17 | Coherence=-116.96 | Top words= coal pay stop line suppoing adanis money labor greens alp
Topic 18 | Coherence=-79.94 | Top words= queensland coal decide loan ll scrubbed independent director exclusive page
Topic 19 | Coherence=-55.14 | Top words= farmers advice coonamble week gas people lied tyres access away
Time taken to process dataset: 45.658793687820435 seconds, 0.7609798947970072 minutes, 0.012682998246616787 hours.

Process finished with exit code 0

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
import pyLDAvis
from biterm.cbtm import oBTM
from biterm.utility import vec_to_biterms, topic_summuary  # helper functions

# Import custom utility functions.
# import topic_extraction_utility_functions as lda_util

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


################################################################################################################

def biterm_topic_model_topic_extraction():
    """
    Function performs topic extraction on Tweets using the Gensim HDP model.

    :return: None.
    """
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(slo_feature_series)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # log.info(f"\n.fit_transform - Learn the vocabulary dictionary and return term-document matrix.")
    # log.info(f"{tf}\n")
    # log.info(f"\n.get_feature_names - Array mapping from feature integer indices to feature name")
    # log.info(f"{tf_feature_names}\n")

    log.info("\n.fit_transform - Learn the vocabulary dictionary and return term-document matrix.")
    log.info(tf)
    log.info("\n.get_feature_names - Array mapping from feature integer indices to feature name")
    log.info(tf_feature_names)

    # Convert corpus of documents (vectorized text) to numpy array.
    tf_array = tf.toarray()

    # Convert dictionary of words (vocabulary) to numpy array.
    tf_feature_names = np.array(tf_vectorizer.get_feature_names())

    # get biterms
    biterms = vec_to_biterms(tf_array)

    # create btm
    btm = oBTM(num_topics=20, V=tf_feature_names)

    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 100):  # prozess chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=50)
    topics = btm.transform(biterms)
    time.sleep(3)

    # print("\n\n Visualize Topics ..")
    # vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(tf_array, axis=1), tf_feature_names, np.sum(tf_array, axis=0))
    # pyLDAvis.save_html(vis, './vis/online_btm.html')

    print("\n\n Topic coherence ..")
    topic_summuary(btm.phi_wz.T, tf_array, tf_feature_names, 10)

    print("\n\n Texts & Topics ..")
    for i in range(len(slo_feature_series)):
        print("{} (topic: {})".format(slo_feature_series[i], topics[i].argmax()))

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
    biterm_topic_model_topic_extraction()

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
