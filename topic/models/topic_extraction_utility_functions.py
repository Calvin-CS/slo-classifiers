"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 7-9-19

SLO Topic Extraction Utility Functions.

###########################################################
Notes:

Adapted code from SLO TBL Topic Classification code-base for use in Tweet pre-processing.

Regex to replace f-strings:

Find:

(print\()f(.*)\{(.*)\}(.*"\))

Replace:

$1$2" +$3+ "$4

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

https://spacy.io/usage/spacy-101
https://www.datacamp.com/community/blog/spacy-cheatsheet
https://spacy.io/usage/processing-pipelines#disabling
https://www.nltk.org/api/nltk.tokenize.html

"""

################################################################################################################
################################################################################################################

# Import libraries.
import html
import logging as log
import re
import string
import time
import warnings
import pandas as pd
import spacy
import nltk
from nltk import WordNetLemmatizer, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
# Import custom utility functions.
import slo_twitter_data_analysis_utility_functions as tweet_util_v2

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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
log.disable(level=log.DEBUG)

"""
Modify spaCy Pipeline for tokenization.
"""
nlp = spacy.load('en')
nlp.remove_pipe("parser")
nlp.remove_pipe("tagger")
nlp.remove_pipe("ner")


################################################################################################################
################################################################################################################

def preprocess_tweet_text(tweet_text):
    """
    Helper function performs text pre-processing using regular expressions and other Python functions.

    Resources:

    https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/settings.py

    :return: the processed text.
    """
    # Check that there is text, otherwise convert to empty string.
    if type(tweet_text) == float:
        tweet_text = ""

    # Convert html chars. to unicode chars.
    tweet_text = html.unescape(tweet_text)

    # Remove "RT" tags.
    preprocessed_tweet_text = re.sub(r'^(RT @\w+: )', "", tweet_text)

    # Remove concatenated URL's.
    preprocessed_tweet_text = re.sub(r'(.)http', r'\1 http', preprocessed_tweet_text)

    # Handle whitespaces.
    preprocessed_tweet_text = re.sub(r'\s', " ", preprocessed_tweet_text)

    # Remove URL's.
    preprocessed_tweet_text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                     r"slo_url", preprocessed_tweet_text)

    # Remove Tweet mentions.
    preprocessed_tweet_text = re.sub(r'@[a-zA-Z_0-9]+', r"slo_mention", preprocessed_tweet_text)

    # Remove Tweet stock symbols.
    preprocessed_tweet_text = re.sub(r'$[a-zA-Z]+', r"slo_stock", preprocessed_tweet_text)

    # Remove Tweet hashtags.
    preprocessed_tweet_text = re.sub(r'#\w+', r"slo_hash", preprocessed_tweet_text)

    # Remove Tweet cashtags.
    preprocessed_tweet_text = \
        re.sub(r'\$(?=\(.*\)|[^()]*$)\(?\d{1,3}(,?\d{3})?(\.\d\d?)?\)?([bmk]| hundred| thousand| million| billion)?',
               r"slo_cash", preprocessed_tweet_text)

    # Remove Tweet year.
    preprocessed_tweet_text = re.sub(r'[12][0-9]{3}', r"slo_year", preprocessed_tweet_text)

    # Remove Tweet time.
    preprocessed_tweet_text = re.sub(r'[012]?[0-9]:[0-5][0-9]', r"slo_time", preprocessed_tweet_text)

    # Remove character elongations.
    preprocessed_tweet_text = re.sub(r'(.)\1{2,}', r'\1\1\1', preprocessed_tweet_text)

    # Do not remove anything.
    delete_list = []

    # Convert series to string.
    tweet_string = str(preprocessed_tweet_text)

    # Split Tweet into individual words using Python (tokenize)
    # individual_words = tweet_string.split()

    # Tokenize using nltk.
    # nltk_tweet_tokenizer = TweetTokenizer()
    # individual_words = nltk_tweet_tokenizer.tokenize(tweet_string)
    # individual_words = word_tokenize(tweet_string)

    # Tokenize using spacy.
    individual_words = nlp(tweet_string)

    # Check to see if a word is irrelevant or not.
    words_relevant = []
    for w in individual_words:
        if w.text not in delete_list:
            words_relevant.append(w.text)

    # Convert list back into original Tweet text minus irrelevant words.
    tweet_string = ' '.join(words_relevant)
    # Convert back to a series object.
    tweet_series = pd.Series(tweet_string)

    log.debug("Tweet text preprocessed: ")
    # log.debug(f"{tweet_series}\n")
    log.debug(tweet_series)
    log.debug("\n")

    return tweet_series


################################################################################################################

def postprocess_tweet_text(tweet_text):
    """
    Helper function performs text post-processing using regular expressions and other Python functions.

    Resources:

    https://github.com/Calvin-CS/slo-classifiers/blob/master/stance/data/settings.py
    https://spacy.io/api/token

    :return: the processed text.
    """
    # Remove all punctuation. (using spaCy check instead)
    # postprocessed_tweet_text = tweet_text.translate(str.maketrans('', '', string.punctuation))

    # Remove all "slo_" placeholders. TODO - make functional
    # preprocessed_tweet_text = re.sub(r'(.*slo_.*)', r" ", tweet_text)

    # Remove irrelevant words from Tweets.
    delete_list = ["word_n", "woodside", "auspol", "adani",
                   "stopadani",
                   "ausbiz", "santos", "whitehaven", "tinto", "fortescue", "bhp", "adelaide", "billiton", "csg",
                   "nswpol",
                   "nsw", "lng", "don", "rio", "pilliga", "australia", "asx", "just", "today", "great", "says", "like",
                   "big", "better", "rite", "would", "SCREEN_NAME", "mining", "former", "qldpod", "qldpol", "qld", "wr",
                   "melbourne", "andrew", "fuck", "spadani", "greg", "th", "australians", "http", "https", "rt",
                   "goadani",
                   "co", "amp", "riotinto", "carmichael", "abbot", "bill shorten",
                   "slo_url", "slo_mention", "slo_hash", "slo_year", "slo_time", "slo_cash", "slo_stock"]

    # Remove stop words from Tweets using nltk.
    # delete_list = list(stopwords.words('english'))

    # Do not remove anything.
    # delete_list = []

    # Convert series to string.
    tweet_string = str(tweet_text)

    # Split Tweet into individual words using Python (tokenize)
    # individual_words = tweet_string.split()

    # Tokenize using nltk.
    # nltk_tweet_tokenizer = TweetTokenizer()
    # individual_words = nltk_tweet_tokenizer.tokenize(tweet_string)
    # individual_words = word_tokenize(tweet_string)

    # Tokenize using spacy.
    individual_words = nlp(tweet_string)

    # NLTK word lemmatizer.
    # lemmatizer = WordNetLemmatizer()

    # Check to see if a word is irrelevant or not.
    words_relevant = []
    for w in individual_words:
        # If not a irrelevant word and not punctuation and a stop word.
        if w.text not in delete_list and not w.is_punct and not w.is_stop:
            # word_lemmatized = lemmatizer.lemmatize(w)
            # words_relevant.append(word_lemmatized)
            words_relevant.append(w.lemma_.lower())
        else:
            # log.debug(f"Irrelevant word found: {w.text}\n")
            log.debug("Irrelevant word found: " + str(w.text) + "\n")

    # Convert list back into original Tweet text minus irrelevant words.
    tweet_string = ' '.join(words_relevant)
    # Convert back to a series object.
    tweet_series = pd.Series(tweet_string)

    log.debug("Tweet text postprocessed: ")
    # log.debug(f"{tweet_series}\n")
    log.debug(tweet_series)
    log.debug("\n")

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
        pd.read_csv(input_file_path, sep=",", encoding="utf-8")

    # Generate a Pandas dataframe.
    twitter_dataframe = pd.DataFrame(twitter_dataset[column_name])

    # Print shape and column names.
    log.info("\nThe shape of our raw unpreprocessed text:")
    log.info(twitter_dataframe.shape)
    log.info("\nThe columns of our unpreprocessed text:")
    log.info(twitter_dataframe.columns)
    log.info("\nA sample of our unpreprocessed text:")
    log.info(twitter_dataframe.sample(1))

    #######################################################

    # # Down-case all text. (using spacy instead now - deprecated)
    # twitter_dataframe[f"{column_name}"] = twitter_dataframe[column_name].str.lower()

    # # Pre-process each tweet individually (calls a helper function).
    # twitter_dataframe[f"{column_name}_preprocessed"] = twitter_dataframe[f"{column_name}"].apply(preprocess_tweet_text)
    #
    # # Post-process each tweet individually (calls a helper function).
    # twitter_dataframe[f"{column_name}_postprocessed"] = \
    #     twitter_dataframe[f"{column_name}_preprocessed"].apply(postprocess_tweet_text)

    # Pre-process each tweet individually (calls a helper function).
    twitter_dataframe[column_name + "_preprocessed"] = twitter_dataframe[column_name].apply(preprocess_tweet_text)

    # Post-process each tweet individually (calls a helper function).
    twitter_dataframe[column_name + "_postprocessed"] = \
        twitter_dataframe[column_name + "_preprocessed"].apply(postprocess_tweet_text)

    # Reindex everything.
    twitter_dataframe.index = pd.RangeIndex(len(twitter_dataframe.index))

    # # Save to CSV file.
    # twitter_dataframe.to_csv(f"{output_file_path}", sep=',',
    #                          encoding='utf-8', index=False)

    # Save to CSV file.
    twitter_dataframe.to_csv(output_file_path, sep=',',
                             encoding='utf-8', index=False)

    # print(f"\nDataset preprocessing finished.")
    # print(f"Saved to: {output_file_path}\n")

    print("\nDataset preprocessing finished.")
    print("Saved to: " + output_file_path + "\n")


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

def non_negative_matrix_factorization_grid_search(dataframe, search_parameters):
    """
    Function performs exhaustive grid search for Scikit-Learn NMF model.

    https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    https://stackoverflow.com/questions/54983241/gridsearchcv-and-randomizedsearchcv-sklearn-typeerror-call-missing-1-r
    https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning/44661188

    FIXME - no functional unless we find a way to disable cross-validation since NMF is unsupervised training.
    :return: None.
    """
    from sklearn.decomposition import NMF
    from sklearn.model_selection import GridSearchCV

    # Construct the pipeline.
    non_negative_matrix_factorization_clf = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', NMF()),
    ])

    # Perform the grid search.
    non_negative_matrix_factorization_clf = GridSearchCV(non_negative_matrix_factorization_clf, search_parameters, cv=[(slice(None), slice(None))],
                                                         iid=False, n_jobs=None, scoring="accuracy")
    non_negative_matrix_factorization_clf.fit(dataframe)

    # View all the information stored in the model after training it.
    classifier_results = pd.DataFrame(non_negative_matrix_factorization_clf.cv_results_)
    log.debug("The shape of the Non-Negative Matrix Factorization model's result data structure is:")
    log.debug(classifier_results.shape)
    log.debug(
        "The contents of the Non-Negative Matrix Factorization model's result data structure is:")
    log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.info("The optimal parameters found for the Non-Negative Matrix Factorization is:")
    for param_name in sorted(search_parameters.keys()):
        log.info("%s: %r" % (param_name, non_negative_matrix_factorization_clf.best_params_[param_name]))
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
    tweet_dataframe_processed_column_names = ['text_derived', 'text_derived_preprocessed', 'text_derived_postprocessed']

    # Rename column in dataframe.
    tweet_dataframe_processed.columns = tweet_dataframe_processed_column_names

    # Create input feature.
    selected_features = tweet_dataframe_processed[tweet_dataframe_processed_column_names]
    processed_features = selected_features.copy()

    # Create feature set.
    slo_feature_set = processed_features['text_derived_postprocessed']

    return slo_feature_set


################################################################################################################

def topic_author_model_group_by_dataset_row_index_value(tweet_dataframe, debug_boolean):
    """
    Function to combine all Tweets by the same author into one document (example) for topic extraction.

    Important Note:

    Row index values are always two lower than the actual row index values in the CSV dataset file.
    This may have something to do with how Pandas dataframe is generated from the CSV dataset file.
    Therefore, hack-fix by incrementing by a +2 each time (ONLY IF necessary; however, it could be correctly associated
    in the Pandas dataframe itself, just not to the CSV itself since that has a header row with column names; however,
    this doesn't explain why it is two off rather than just one off due to the header row with column names)

    Resources:

    (below lsit of URL's for grouping all Tweets by common author)
    https://stackoverflow.com/questions/47434426/pandas-groupby-unique-multiple-columns
    https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

    (below list of URL"s for converting from dataframe to dictionary of key: author, value: tweet ID's)
    https://stackoverflow.com/questions/18012505/python-pandas-dataframe-columns-convert-to-dict-key-and-value
    https://stackoverflow.com/questions/18695605/python-pandas-dataframe-to-dictionary
    https://www.geeksforgeeks.org/zip-in-python/
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html

    https://stackoverflow.com/questions/43193880/how-to-get-row-number-in-dataframe-in-pandas

    :param debug_boolean: turn debug export to CSV on or off.
    :param tweet_dataframe: Pandas dataframe containing Twitter dataset.
    :return: None.
    """
    tweet_dataframe = pd.DataFrame(tweet_dataframe)

    ##########################################################

    # # Group Tweets by Author with all Tweet fields included.
    # group_by_authors_with_all_attributes = tweet_dataframe.groupby(["user_screen_name"])
    # group_by_authors_with_all_attributes = pd.DataFrame(group_by_authors_with_all_attributes)
    #
    # group_by_authors_with_all_attributes.columns = ["user_screen_name", "all_attributes"]
    # print(f"Dataframe columns:\n {group_by_authors_with_all_attributes.columns}\n")
    # print(f"Dataframe shape:\n {group_by_authors_with_all_attributes.shape}\n")
    # print(f"Dataframe samples:\n {group_by_authors_with_all_attributes.sample(5)}\n")
    #
    # for element in group_by_authors_with_all_attributes["all_attributes"]:
    #     row_index_and_tweet_id_to_list = element["tweet_id"].to_string().split()
    #     row_index_value_as_integer = int(row_index_and_tweet_id_to_list[0])
    #     tweet_id_as_integer = int(float(row_index_and_tweet_id_to_list[1]))
    #     print(f"Tweet row index value in dataset: {row_index_value_as_integer}")
    #     print(f"Tweet ID: {tweet_id_as_integer}")
    # #
    # group_by_author_with_all_attributes_dictionary = \
    #     dict(zip(group_by_authors_with_all_attributes["user_screen_name"],
    #              group_by_authors_with_all_attributes["all_attributes"]))

    # group_by_authors_with_all_attributes = group_by_authors_with_all_attributes.to_dict("records")

    # for element in group_by_authors_with_all_attributes:
    #     print(f"Nested dictionary in list:\n {element}")

    # for key, value in group_by_authors_with_all_attributes.items():
    #     print(f"Author: {key}")
    #     print(f"Associated Tweets (ID's): {value}")

    ##########################################################

    # Group Tweets by Author with Tweet ID field included.
    group_by_author_with_row_index_value = tweet_dataframe.groupby(["user_screen_name"],
                                                                   group_keys=True, as_index=True)
    group_by_author_with_row_index_value = pd.DataFrame(group_by_author_with_row_index_value)

    group_by_author_with_row_index_value.columns = ["user_screen_name", "all_attributes"]
    # print(f"Dataframe columns:\n {group_by_author_with_row_index_value.columns}\n")
    # print(f"Dataframe shape:\n {group_by_author_with_row_index_value.shape}\n")
    # print(f"Dataframe samples:\n {group_by_author_with_row_index_value.sample(5)}\n")
    print("Dataframe columns:\n" + str(group_by_author_with_row_index_value.columns) + "\n")
    print("Dataframe shape:\n" + str(group_by_author_with_row_index_value.shape) + "\n")
    print("Dataframe samples:\n" + str(group_by_author_with_row_index_value.sample(5)) + "\n")

    group_by_author_with_row_index_value_dictionary = {}

    def create_mappings(row):
        """
        Function to create author to Row Index Values mappings.
        :param row: example to operate on.
        :return: append to Dictionary - key: author, value: Row Index Values
        """
        indices = tweet_dataframe.index[tweet_dataframe.user_screen_name == str(row["user_screen_name"])]

        group_by_author_with_row_index_value_dictionary[row["user_screen_name"]] = list(indices)

        row["associated_row_index_values"] = list(indices)
        return row["associated_row_index_values"]

    group_by_author_with_row_index_value["associated_row_index_values"] = \
        group_by_author_with_row_index_value.apply(create_mappings, axis=1)

    # for key, value in group_by_author_with_row_index_value_dictionary.items():
    #     # print(f"Author: {key}")
    #     # print(f"List of associated Row Index Values:\n {value}")
    #     print("Author:" + str(key))
    #     print("List of associated Tweet Texts (documents):\n" + str(value))

    if debug_boolean:
        tweet_util_v2.export_to_csv_json(
            group_by_author_with_row_index_value, ["user_screen_name", "associated_row_index_values"],
            "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
            "group-by-authors-with-row-index-value", "w", "csv")
        # tweet_util_v2.export_to_csv_json(
        #     group_by_author_with_row_index_value, ["user_screen_name", "associated_row_index_values"],
        #     "/home/jj47/Summer-Research-2019-master/"
        #     "group-by-authors-with-row-index-value", "w", "csv")

    return group_by_author_with_row_index_value_dictionary


################################################################################################################

def topic_author_model_group_by_author_tweet_id(tweet_dataframe, debug_boolean):
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
    """
    tweet_dataframe = pd.DataFrame(tweet_dataframe)

    # Group Tweets by Author with Tweet ID field included.
    group_by_author_with_tweet_id = tweet_dataframe.groupby(["user_screen_name"],
                                                            group_keys=True, as_index=True)["tweet_id"]
    group_by_author_with_tweet_id = pd.DataFrame(group_by_author_with_tweet_id)

    group_by_author_with_tweet_id.columns = ["user_screen_name", "associated_tweet_ids"]
    # print(f"Dataframe columns:\n {group_by_author_with_tweet_id.columns}\n")
    # print(f"Dataframe shape:\n {group_by_author_with_tweet_id.shape}\n")
    # print(f"Dataframe samples:\n {group_by_author_with_tweet_id.sample(5)}\n")
    print("Dataframe columns:\n" + str(group_by_author_with_tweet_id.columns) + "\n")
    print("Dataframe shape:\n" + str(group_by_author_with_tweet_id.shape) + "\n")
    print("Dataframe samples:\n" + str(group_by_author_with_tweet_id.sample(5)) + "\n")

    def convert_to_integers(row):
        """
        Function to convert Tweet ID's from scientific notation float to integers.

        :param row: example to operate on.
        :return: Tweet ID's as integers.
        """
        integer_list = []
        for element in row["associated_tweet_ids"]:
            integer_list.append(int(element))
        row["associated_tweet_ids"] = integer_list
        return row["associated_tweet_ids"]

    group_by_author_with_tweet_id_dictionary = {}

    def create_mappings(row):
        """
        Function to create author to tweet ID mappings.
        :param row: example to operate on.
        :return: append to Dictionary - key: author, value: Tweet ID's
        """
        group_by_author_with_tweet_id_dictionary[row["user_screen_name"]] = row["associated_tweet_ids"]

    group_by_author_with_tweet_id["associated_tweet_ids"] = \
        group_by_author_with_tweet_id.apply(convert_to_integers, axis=1)
    group_by_author_with_tweet_id.apply(create_mappings, axis=1)

    # for key, value in group_by_author_with_tweet_id_dictionary.items():
    #     # print(f"Author: {key}")
    #     # print(f"List of associated Tweet ID's:\n {value}")
    #     print("Author:" + str(key))
    #     print("List of associated Tweet Texts (documents):\n" + str(value))

    if debug_boolean:
        tweet_util_v2.export_to_csv_json(
            group_by_author_with_tweet_id, [],
            "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
            "group-by-authors-with-tweet-id", "w", "csv")

    return group_by_author_with_tweet_id_dictionary


################################################################################################################

def topic_author_model_group_by_author_tweet_text(tweet_dataframe, debug_boolean):
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
    """
    tweet_dataframe = pd.DataFrame(tweet_dataframe)

    # Group Tweets by Author with Tweet ID field included.
    group_by_author_with_tweet_text = tweet_dataframe.groupby(["user_screen_name"],
                                                              group_keys=True, as_index=True)["text_derived"]
    group_by_author_with_tweet_text = pd.DataFrame(group_by_author_with_tweet_text)

    group_by_author_with_tweet_text.columns = ["user_screen_name", "associated_tweet_text"]
    # print(f"Dataframe columns:\n {group_by_author_with_tweet_text.columns}\n")
    # print(f"Dataframe shape:\n {group_by_author_with_tweet_text.shape}\n")
    # print(f"Dataframe samples:\n {group_by_author_with_tweet_text.sample(5)}\n")
    print("Dataframe columns:\n" + str(group_by_author_with_tweet_text.columns) + "\n")
    print("Dataframe shape:\n" + str(group_by_author_with_tweet_text.shape) + "\n")
    print("Dataframe samples:\n" + str(group_by_author_with_tweet_text.sample(5)) + "\n")

    def convert_to_strings(row):
        """
        Function to convert Tweet Texts (documents) to strings.

        :param row: example to operate on.
        :return: Tweet Texts (documents) as strings.
        """
        text_list = []
        for element in row["associated_tweet_text"]:
            text_list.append(str(element))
        row["associated_tweet_text"] = text_list
        return row["associated_tweet_text"]

    group_by_author_with_tweet_text_dictionary = {}

    def create_mappings(row):
        """
        Function to create author to Tweet Texts (documents) mappings.
        :param row: example to operate on.
        :return: append to Dictionary - key: author, value: Tweet Texts (documents)
        """
        group_by_author_with_tweet_text_dictionary[row["user_screen_name"]] = row["associated_tweet_text"]

    group_by_author_with_tweet_text["associated_tweet_text"] = \
        group_by_author_with_tweet_text.apply(convert_to_strings, axis=1)
    group_by_author_with_tweet_text.apply(create_mappings, axis=1)

    # for key, value in group_by_author_with_tweet_text_dictionary.items():
    #     # print(f"Author: {key}")
    #     # print(f"List of associated Tweet Texts (documents):\n {value}")
    #     print("Author:" + str(key))
    #     print("List of associated Tweet Texts (documents):\n" + str(value))

    if debug_boolean:
        tweet_util_v2.export_to_csv_json(
            group_by_author_with_tweet_text, [],
            "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
            "group-by-authors-with-tweet-text", "w", "csv")

    return group_by_author_with_tweet_text_dictionary


################################################################################################################


start_time = time.time()

# # Import CSV dataset and convert to dataframe.
# tweet_csv_dataframe = tweet_util_v2.import_dataset(
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "twitter-dataset-7-10-19-test-subset-100-examples.csv",
#     "csv", False)

# Create author-topic model dataframe.
# topic_author_model_group_by_author_tweet_id(tweet_csv_dataframe, False)
# topic_author_model_group_by_author_tweet_text(tweet_csv_dataframe, True)
# topic_author_model_group_by_dataset_row_index_value(tweet_csv_dataframe, True)

# # Test on the already tokenized dataset from stance detection.
# tweet_dataset_preprocessor(
#     "D:/Dropbox/summer-research-2019/datasets/dataset_20100101-20180510_tok_PROCESSED_shortened.csv",
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/lda-ready-test.csv",
#     "tweet_t")

############################################################

# # Test on our topic modeling dataset.
# tweet_dataset_preprocessor(
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "twitter-dataset-7-10-19-test-subset-100-examples.csv",
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "twitter-dataset-7-10-19-topic-extraction-ready-tweet-text-with-hashtags-excluded-created-7-18-19-test.csv",
#     "text_derived")

# # Test on our topic modeling dataset.
# tweet_dataset_preprocessor(
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "twitter-dataset-7-10-19-test-subset-100-examples.csv",
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "twitter-dataset-7-10-19-topic-extraction-ready-user-description-text-with-hashtags-excluded-created-7-18-19-test.csv",
#     "user_description")

############################################################

# # Tokenize using our Twitter dataset.
# tweet_dataset_preprocessor(
#     "/home/jj47/Summer-Research-2019-master/"
#     "twitter-dataset-7-19-19-with-irrelevant-tweets-excluded",
#     "/home/jj47/Summer-Research-2019-master/"
#     "twitter-dataset-7-19-19-topic-extraction-ready-tweet-text-with-hashtags-excluded-created-7-19-19.csv",
#     "text_derived")

# # Tokenize using our Twitter dataset.
# tweet_dataset_preprocessor(
#     "/home/jj47/Summer-Research-2019-master/"
#     "twitter-dataset-7-19-19-with-irrelevant-tweets-excluded",
#     "/home/jj47/Summer-Research-2019-master/"
#     "twitter-dataset-7-19-19-topic-extraction-ready-user-description-text-with-hashtags-excluded-created-7-19-19.csv",
#     "user_description")

############################################################

# end_time = time.time()
# time_elapsed_seconds = end_time - start_time
# time_elapsed_minutes = (end_time - start_time) / 60.0
# time_elapsed_hours = (end_time - start_time) / 60.0 / 60.0
# time.sleep(3)
# log.info(f"The time taken to visualize the statistics is {time_elapsed_seconds} seconds, "
#          f"{time_elapsed_minutes} minutes, {time_elapsed_hours} hours")

end_time = time.time()
time_elapsed_seconds = end_time - start_time
time_elapsed_minutes = (end_time - start_time) / 60.0
time_elapsed_hours = (end_time - start_time) / 60.0 / 60.0
log.info("The time taken to process the file(s) is " + str(time_elapsed_seconds) + "seconds, " +
         str(time_elapsed_minutes) + " minutes, " + str(time_elapsed_hours) + " hours")
