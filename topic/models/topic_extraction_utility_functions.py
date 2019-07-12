"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-9-19

SLO Topic Extraction Utility Functions.

###########################################################
Notes:

Adapted code from SLO TBL Topic Classification code-base for use in Tweet pre-processing.

TODO - implement further elements of post-processing to replicate what is done using CMU Tweet Tagger.

###########################################################
Resources Used:

https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/tweet_preprocessor.py
https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/dataset_processor.py
https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/settings.py
https://github.com/Calvin-CS/slo-classifiers/blob/topic/topic/final_derek/Preprocessing.py

https://www.machinelearningplus.com/nlp/gensim-tutorial/
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

https://www.guru99.com/tokenize-words-sentences-nltk.html
https://radimrehurek.com/gensim/utils.html

https://www.programcreek.com/python/example/98657/nltk.corpus.stopwords.words
https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import re
import string
import warnings
import pandas as pd
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
# Import custom utility functions.
import slo_twitter_data_analysis_utility_functions as tweet_util_v2

nltk.download('stopwords')
nltk.download('wordnet')

#############################################################

# Miscellaneous parameter adjustments for pandas and python.
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

"""
Turn debug log statements for various sections of code on/off.
(adjust log level as necessary)
"""
log.basicConfig(level=log.INFO)


################################################################################################################
################################################################################################################

def preprocess_tweet_text(tweet_text):
    """
    Helper function performs text pre-processing using regular expressions and other Python functions.

    Resources:

    https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/settings.py

    :return: the processed text.
    """
    # Remove "RT" tags.
    preprocessed_tweet_text = re.sub('^(rt @\w+: )', "", tweet_text)

    # Remove URL's.
    preprocessed_tweet_text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                     "", preprocessed_tweet_text)

    # Remove concatenated URL's.
    preprocessed_tweet_text = re.sub('(.)http', "", preprocessed_tweet_text)

    # Remove Tweet mentions.
    preprocessed_tweet_text = re.sub('@[a-zA-Z_0-9]+', "", preprocessed_tweet_text)

    # Remove Tweet stock symbols.
    preprocessed_tweet_text = re.sub('$[a-zA-Z]+', "", preprocessed_tweet_text)

    # Remove Tweet hashtags.
    preprocessed_tweet_text = re.sub('#\w+', "", preprocessed_tweet_text)

    # Remove Tweet cashtags.
    preprocessed_tweet_text = \
        re.sub('\$(?=\(.*\)|[^()]*$)\(?\d{1,3}(,?\d{3})?(\.\d\d?)?\)?([bmk]| hundred| thousand| million| billion)?',
               "", preprocessed_tweet_text)

    # Remove Tweet year.
    preprocessed_tweet_text = re.sub('[12][0-9]{3}', "", preprocessed_tweet_text)

    # Remove Tweet time.
    preprocessed_tweet_text = re.sub('[012]?[0-9]:[0-5][0-9]', "", preprocessed_tweet_text)

    # Remove character elongations.
    preprocessed_tweet_text = re.sub('(.)\1{2,}', "", preprocessed_tweet_text)

    # Remove irrelevant words from Tweets.
    delete_list = ["slo_url", "slo_mention", "word_n", "slo_year", "slo_cash", "woodside", "auspol", "adani",
                   "stopadani",
                   "ausbiz", "santos", "whitehaven", "tinto", "fortescue", "bhp", "adelaide", "billiton", "csg",
                   "nswpol",
                   "nsw", "lng", "don", "rio", "pilliga", "australia", "asx", "just", "today", "great", "says", "like",
                   "big", "better", "rite", "would", "SCREEN_NAME", "mining", "former", "qldpod", "qldpol", "qld", "wr",
                   "melbourne", "andrew", "fuck", "spadani", "greg", "th", "australians", "http", "https", "rt",
                   "goadani",
                   "co", "amp", "riotinto", "carmichael", "abbot", "bill shorten",
                   "slourl", "slomention", "slohashtag", "sloyear", "slocash"]

    # Convert series to string.
    tweet_string = str(preprocessed_tweet_text)

    # Split Tweet into individual words.
    individual_words = tweet_string.split()

    # Check to see if a word is irrelevant or not.
    words_relevant = []
    for w in individual_words:
        if w not in delete_list:
            words_relevant.append(w)
        else:
            log.debug(f"Irrelevant word found: {w}\n")

    # Convert list back into original Tweet text minus irrelevant words.
    tweet_string = ' '.join(words_relevant)
    # Convert back to a series object.
    tweet_series = pd.Series(tweet_string)

    log.debug("Tweet text with irrelevant words removed: ")
    log.debug(f"{tweet_series}\n")

    return tweet_series


################################################################################################################

def postprocess_tweet_text(tweet_text):
    """
    Helper function performs text post-processing using regular expressions and other Python functions.

    Resources:

    https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/settings.py

    :return: the processed text.
    """
    # Remove all punctuation.
    postprocessed_tweet_text = tweet_text.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words from Tweets.
    delete_list = list(stopwords.words('english'))

    # Convert series to string.
    tweet_string = str(postprocessed_tweet_text)

    # Split Tweet into individual words.
    individual_words = tweet_string.split()

    # NLTK word lemmatizer.
    lemmatizer = WordNetLemmatizer()

    # Check to see if a word is irrelevant or not.
    words_relevant = []
    for w in individual_words:
        if w not in delete_list:
            word_lemmatized = lemmatizer.lemmatize(w)
            words_relevant.append(word_lemmatized)

    # Convert list back into original Tweet text minus irrelevant words.
    tweet_string = ' '.join(words_relevant)
    # Convert back to a series object.
    tweet_series = pd.Series(tweet_string)

    log.debug("Tweet text with stop words removed and lemmatized words: ")
    log.debug(f"{tweet_series}\n")

    return tweet_series


################################################################################################################

def tweet_dataset_preprocessor(input_file_path, output_file_path, column_name):
    """
     Function pre-processes specified dataset in preparation for LDA topic extraction.

    :param input_file_path: relative filepath from project root directory for location of dataset to process.
    :param output_file_path: relative filepath from project root directory for location to save .csv file.
    :param column_name: name of the column in the dataset that we are pre-processing.
    :return: Nothing. Saves to CSV file.
    """

    # Import the dataset.
    twitter_dataset = \
        pd.read_csv(f"{input_file_path}", sep=",", encoding="utf-8")

    # Generate a Pandas dataframe.
    twitter_dataframe = pd.DataFrame(twitter_dataset[f"{column_name}"])

    # Print shape and column names.
    log.info(f"\nThe shape of our raw unpreprocessed text:")
    log.info(twitter_dataframe.shape)
    log.info(f"\nThe columns of our unpreprocessed text:")
    log.info(twitter_dataframe.head)

    #######################################################

    # Down-case all text.
    twitter_dataframe[f"{column_name}"] = twitter_dataframe[column_name].str.lower()

    # Pre-process each tweet individually (calls a helper function).
    twitter_dataframe[f"{column_name}_preprocessed"] = twitter_dataframe[f"{column_name}"].apply(preprocess_tweet_text)

    # Post-process each tweet individuall (calls a helper function).
    twitter_dataframe[f"{column_name}_postprocessed"] = \
        twitter_dataframe[f"{column_name}_preprocessed"].apply(postprocess_tweet_text)

    # Reindex everything.
    twitter_dataframe.index = pd.RangeIndex(len(twitter_dataframe.index))

    # Save to CSV file.
    twitter_dataframe.to_csv(f"{output_file_path}", sep=',',
                             encoding='utf-8', index=False)

    print(f"\nDataset preprocessing finished.")
    print(f"Saved to: {output_file_path}\n")


################################################################################################################

def display_topics(model, feature_names, num_top_words):
    """
    Helper function to display the top words for each topic in the LDA model.

    :param model: the LDA model
    :param feature_names: feature names from CountVectorizer.
    :param num_top_words: # of words to display for each topic.
    :return: None.
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))


################################################################################################################

def latent_dirichlet_allocation_grid_search(dataframe, search_parameters):
    """
    Function performs exhaustive grid search for Scikit-Learn LDA model.

    :return: None.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.model_selection import GridSearchCV

    # Construct the pipeline.
    latent_dirichlet_allocation_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LatentDirichletAllocation()),
    ])

    # Perform the grid search.
    latent_dirichlet_allocation_clf = GridSearchCV(latent_dirichlet_allocation_clf, search_parameters, cv=5,
                                                   iid=False, n_jobs=-1)
    latent_dirichlet_allocation_clf.fit(dataframe)

    # View all the information stored in the model after training it.
    classifier_results = pd.DataFrame(latent_dirichlet_allocation_clf.cv_results_)
    log.debug("The shape of the Latent Dirichlet Allocation model's result data structure is:")
    log.debug(classifier_results.shape)
    log.debug(
        "The contents of the Latent Dirichlet Allocation model's result data structure is:")
    log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.info("The optimal parameters found for the Latent Dirichlet Allocation is:")
    for param_name in sorted(search_parameters.keys()):
        log.info("%s: %r" % (param_name, latent_dirichlet_allocation_clf.best_params_[param_name]))
    log.info("\n")


################################################################################################################

def dataframe_subset(tweet_dataset, sample_size):
    """
    Function slices the Twitter dataset into a smaller dataset for the purposes of saving compute time on using
    exhaustive grid search to find initial optimal hyper parameters for LDA topic modeling and extraction.

    :param tweet_dataset: the dataset to generate a subset from.
    :param sample_size: size of the subset to construct.
    :return: feature set to use for GridSearchCV.
    """

    # Check that user passed in dataset from which we can generate a dataframe.
    # noinspection PyBroadException
    try:
        tweet_dataframe_processed = pd.DataFrame(tweet_dataset)
    except Exception:
        print("Failed to generate Dataframe for subsetting operation:")
        return

    # Drop any NaN or empty Tweet rows in dataframe (or else CountVectorizer will blow up).
    tweet_dataframe_processed = tweet_dataframe_processed.dropna()

    # Reindex everything.
    tweet_dataframe_processed.index = pd.RangeIndex(len(tweet_dataframe_processed.index))

    # Subset of the dataframe.
    tweet_dataframe_processed.sample(n=sample_size)

    # Assign column names.
    tweet_dataframe_processed_column_names = ['Tweet']

    # Rename column in dataframe.
    tweet_dataframe_processed.columns = tweet_dataframe_processed_column_names

    # Create input feature.
    selected_features = tweet_dataframe_processed[tweet_dataframe_processed_column_names]
    processed_features = selected_features.copy()

    # Create feature set.
    slo_feature_set = processed_features['Tweet']

    return slo_feature_set


################################################################################################################

def topic_author_model(tweet_dataframe, debug_boolean):
    """
    Function to combine all Tweets by the same author into one document (example) for topic extraction.

    Resources:

    (below lsit of URL's for grouping all Tweets by common author)
    https://stackoverflow.com/questions/47434426/pandas-groupby-unique-multiple-columns
    https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

    (below list of URL"s for converting from dataframe to dictionary of key: author, value: tweet ID's)
    https://stackoverflow.com/questions/18012505/python-pandas-dataframe-columns-convert-to-dict-key-and-value
    https://stackoverflow.com/questions/18695605/python-pandas-dataframe-to-dictionary
    https://www.geeksforgeeks.org/zip-in-python/
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html

    :param debug_boolean: turn debug export to CSV on or off.
    :param tweet_dataframe: Pandas dataframe containing Twitter dataset.
    :return: None.
    TODO - remove multi-index values from the output.
    """
    tweet_dataframe = pd.DataFrame(tweet_dataframe)

    # # Group Tweets by Author with all Tweet fields included.
    # group_by_authors_with_all_attributes = tweet_dataframe.groupby(["user_screen_name"])
    # group_by_authors_with_all_attributes = pd.DataFrame(group_by_authors_with_all_attributes)

    # Group Tweets by Author with Tweet ID field included.
    group_by_author_with_tweet_id = tweet_dataframe.groupby(["user_screen_name"],
                                                            group_keys=True, as_index=True)["tweet_id"]
    group_by_author_with_tweet_id = pd.DataFrame(group_by_author_with_tweet_id)

    group_by_author_with_tweet_id.columns = ["user_screen_name", "associated_tweet_ids"]
    print(f"Dataframe columns:\n {group_by_author_with_tweet_id.columns}\n")
    print(f"Dataframe shape:\n {group_by_author_with_tweet_id.shape}\n")
    print(f"Dataframe samples:\n {group_by_author_with_tweet_id.sample(5)}\n")

    group_by_author_with_tweet_id_dictionary = dict(zip(group_by_author_with_tweet_id["user_screen_name"],
                                                        group_by_author_with_tweet_id["associated_tweet_ids"]))

    # group_by_author_with_tweet_id = group_by_author_with_tweet_id.to_dict("records")

    # for element in group_by_author_with_tweet_id:
    #     print(f"Nested dictionary in list:\n {element}")

    for key, value in group_by_author_with_tweet_id_dictionary.items():
        print(f"Author: {key}")
        print(f"Associated Tweets (ID's): {value}")

    # # Group Tweets by Author with Tweet full text field included.
    # group_by_author_with_tweet_text = tweet_dataframe.groupby(["user_screen_name"])["tweet_full_text"]
    # group_by_author_with_tweet_text = pd.DataFrame(group_by_author_with_tweet_text)

    if debug_boolean:
        # tweet_util_v2.export_to_csv_json(
        #     group_by_authors_with_all_attributes, [],
        #     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
        #     "group-by-authors-with-all-attributes", "w", "csv")

        tweet_util_v2.export_to_csv_json(
            group_by_author_with_tweet_id, [],
            "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
            "group-by-authors-with-tweet-id", "w", "csv")

        # tweet_util_v2.export_to_csv_json(
        #     group_by_author_with_tweet_text, [],
        #     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
        #     "group-by-authors-with-tweet-text", "w", "csv")


################################################################################################################

# Import CSV dataset and convert to dataframe.
tweet_csv_dataframe = tweet_util_v2.import_dataset(
    "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
    "twitter-dataset-6-27-19-test-subset.csv",
    "csv", False)

# Create author-topic model dataframe.
topic_author_model(tweet_csv_dataframe, False)

# # Test on the already tokenized dataset from stance detection.
# tweet_dataset_preprocessor(
#     "D:/Dropbox/summer-research-2019/datasets/dataset_20100101-20180510_tok_PROCESSED_shortened.csv",
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/lda-ready-test.csv",
#     "tweet_t")


# # Test on our topic modeling dataset.
# tweet_dataset_preprocessor(
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/twitter-dataset-6-27-19-test-subset.csv",
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/twitter-dataset-6-27-19-lda-ready-test.csv",
#     "text_derived")
