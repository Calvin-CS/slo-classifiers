"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-23-19

Final Project - SLO TBL Topic Classification

###########################################################
Notes:

These function(s) performs Tweet pre-processing SPECIFIC TO A SINGLE DATASET ONLY.
This is NOT a generalized Tweet dataset preprocessor!!!

###########################################################
Resources Used:

Refer to slo_topic_classification_v1-0.py for a full list of URL's to online resources referenced.

"""

################################################################################################################

import re
import string
import time
import warnings
import pandas as pd
import logging as log

# Note: Need to set level AND turn on debug variables in order to see all debug output.
log.basicConfig(level=log.DEBUG)

# Miscellaneous parameter adjustments for pandas and python.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Turn on and off to debug various sub-sections.
debug = True


################################################################################################################
def preprocess_tweet_text(tweet_text):
    """
    Helper function performs text pre-processing using regular expressions and other Python functions.

    Notes:

    Stop words are retained.

    TODO - shrink character elongations
    TODO - remove non-english tweets
    TODO - remove non-company associated tweets
    TODO - remove year and time.
    TODO - remove cash items?

    :return: the processed Tweet.
    """

    # Remove "RT" tags.
    preprocessed_tweet_text = re.sub("rt", "", tweet_text)

    # Remove URL's.
    preprocessed_tweet_text = re.sub("http[s]?://\S+", "slo_url", preprocessed_tweet_text)

    # Remove Tweet mentions.
    preprocessed_tweet_text = re.sub("@\S+", "slo_mention", preprocessed_tweet_text)

    # Remove Tweet hashtags.
    preprocessed_tweet_text = re.sub("#\S+", "slo_hashtag", preprocessed_tweet_text)

    # Remove all punctuation.
    preprocessed_tweet_text = preprocessed_tweet_text.translate(str.maketrans('', '', string.punctuation))

    return preprocessed_tweet_text


################################################################################################################
def tweet_dataset_preprocessor_1():
    """
    Function pre-processes tbl_training_set.csv in preparation for machine learning input feature creation.

    :return: Nothing. Saves to CSV file.
    """

    # Import the dataset.
    slo_dataset = \
        pd.read_csv("tbl-datasets/tbl_training_set.csv", sep=",")

    # Shuffle the data randomly.
    slo_dataset = slo_dataset.reindex(
        pd.np.random.permutation(slo_dataset.index))

    # Rename columns to something that makes sense.
    column_names = ['Tweet', 'SLO1', 'SLO2', 'SLO3']

    # Generate a Pandas dataframe.
    slo_dataframe = pd.DataFrame(slo_dataset)

    if debug:
        # Print shape and column names.
        log.debug("The shape of our SLO dataframe:")
        log.debug(slo_dataframe.shape)
        log.debug("\n")
        log.debug("The columns of our SLO dataframe:")
        log.debug(slo_dataframe.head)
        log.debug("\n")

    # Assign column names.
    slo_dataframe.columns = column_names

    ################################################################################################################

    # Drop all rows with only NaN in all columns.
    slo_dataframe = slo_dataframe.dropna(how='all')
    # Drop all rows without at least 2 non NaN values - indicating no existing SLO TBL topic classification labels.
    slo_dataframe = slo_dataframe.dropna(thresh=2)

    if debug:
        # Iterate through each row and check we dropped properly.
        log.debug("Dataframe with only examples that have SLO TBL topic classification labels:")
        for index in slo_dataframe.index:
            log.debug(slo_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo_dataframe['SLO1'][index])
                      + '\tSLO2: ' + str(slo_dataframe['SLO2'][index]) + '\tSLO3: ' + str(slo_dataframe['SLO3'][index]))
        log.debug("\n")
        log.debug(
            "Shape of dataframe with only examples that have SLO TBL topic classifications: " + str(
                slo_dataframe.shape))
        log.debug("\n")

    #######################################################

    # Boolean indexing to select examples with only a single SLO TBL topic classification.
    mask = slo_dataframe['SLO1'].notna() & (slo_dataframe['SLO2'].isna() & slo_dataframe['SLO3'].isna())

    if debug:
        # Check that boolean indexing is working.
        log.debug("Check that our boolean indexing mask gives only examples with a single SLO TBL topic "
                  "classification:")
        log.debug(mask.tail)
        log.debug("\n")
        log.debug("The shape of our boolean indexing mask:")
        log.debug(mask.shape)
        log.debug("\n")

    # Create new dataframe with examples that have only a single SLO TBL topic classification.
    slo_dataframe_single_classification = slo_dataframe[mask]

    if debug:
        # Check that we have created the new dataframe properly.
        log.debug("Dataframe with only examples that have a single SLO TBL topic classification labels:")
        # Iterate through each row and check that only examples with a single SLO TBL topic classification are left.
        for index in slo_dataframe_single_classification.index:
            log.debug(slo_dataframe_single_classification['Tweet'][index]
                      + '\tSLO1: ' + str(slo_dataframe_single_classification['SLO1'][index])
                      + '\tSLO2: ' + str(slo_dataframe_single_classification['SLO2'][index])
                      + '\tSLO3: ' + str(slo_dataframe_single_classification['SLO3'][index]))
        log.debug("\n")
        log.debug("Shape of dataframe with only examples that have a single SLO TBL topic classification: "
                  + str(slo_dataframe_single_classification.shape))
        log.debug("\n")

    #######################################################

    # Drop SLO2 and SLO3 columns as they are just NaN values.
    slo_dataframe_single_classification = slo_dataframe_single_classification.drop(columns=['SLO2', 'SLO3'])

    if debug:
        # Check that we have dropped SLO2 and SLO3 columns properly.
        log.debug("Dataframe with SLO2 and SLO3 columns dropped as they are just NaN values:")
        # Iterate through each row and check that each example only has one SLO TBL topic classification left.
        for index in slo_dataframe_single_classification.index:
            log.debug(slo_dataframe_single_classification['Tweet'][index] + '\tSLO1: '
                      + str(slo_dataframe_single_classification['SLO1'][index]))
        log.debug("\n")
        log.debug(
            "Shape of dataframe with SLO2 and SLO3 columns dropped: " + str(slo_dataframe_single_classification.shape))
        log.debug("\n")

    # Re-name columns for clarity of purpose.
    column_names_single = ['Tweet', 'SLO']
    slo_dataframe_single_classification.columns = column_names_single

    #######################################################

    # Boolean indexing to select examples with multiple SLO TBL topic classifications.
    mask = slo_dataframe['SLO1'].notna() & (slo_dataframe['SLO2'].notna() | slo_dataframe['SLO3'].notna())

    if debug:
        # Check that boolean indexing is working.
        log.debug(
            "Check that our boolean indexing mask gives only examples with multiple SLO TBL topic classifications:")
        log.debug(mask.tail)
        log.debug("\n")
        log.debug("The shape of our boolean indexing mask:")
        log.debug(mask.shape)
        log.debug("\n")

    # Create new dataframe with only those examples with multiple SLO TBL topic classifications.
    slo_dataframe_multiple_classifications = slo_dataframe[mask]

    if debug:
        # Check that we have created the new dataframe properly.
        log.debug("Dataframe with only examples that have multiple SLO TBL topic classification labels:")
        # Iterate through each row and check that only examples with multiple SLO TBL topic classifications are left.
        for index in slo_dataframe_multiple_classifications.index:
            log.debug(slo_dataframe_multiple_classifications['Tweet'][index]
                      + '\tSLO1: ' + str(slo_dataframe_multiple_classifications['SLO1'][index])
                      + '\tSLO2: ' + str(slo_dataframe_multiple_classifications['SLO2'][index])
                      + '\tSLO3: ' + str(slo_dataframe_multiple_classifications['SLO3'][index]))
        log.debug("\n")
        log.debug("Shape of dataframe with only examples that have multiple SLO TBL topic classifications: "
                  + str(slo_dataframe_multiple_classifications.shape))
        log.debug("\n")

    #######################################################

    # Duplicate examples with multiple SLO TBL classifications into examples with only 1 SLO TBL topic classification
    # each.
    slo1_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO2', 'SLO3'])
    slo2_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO1', 'SLO3'])
    slo3_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO1', 'SLO2'])

    if debug:
        # Check that we have created the new dataframes properly.
        log.debug(
            "Separated dataframes with only a single label for examples with multiple SLO TBL topic classification "
            "labels:")
        # Iterate through each row and check that each example only has one SLO TBL topic classification left.
        for index in slo1_dataframe.index:
            log.debug(slo1_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo1_dataframe['SLO1'][index]))
        for index in slo2_dataframe.index:
            log.debug(slo2_dataframe['Tweet'][index] + '\tSLO2: ' + str(slo2_dataframe['SLO2'][index]))
        for index in slo3_dataframe.index:
            log.debug(slo3_dataframe['Tweet'][index] + '\tSLO3: ' + str(slo3_dataframe['SLO3'][index]))
        log.debug("\n")
        log.debug("Shape of slo1_dataframe: " + str(slo1_dataframe.shape))
        log.debug("Shape of slo2_dataframe: " + str(slo2_dataframe.shape))
        log.debug("Shape of slo3_dataframe: " + str(slo3_dataframe.shape))
        log.debug("\n")

    # Re-name columns for clarity of purpose.
    column_names_single = ['Tweet', 'SLO']

    slo1_dataframe.columns = column_names_single
    slo2_dataframe.columns = column_names_single
    slo3_dataframe.columns = column_names_single

    #######################################################

    # Concatenate the individual dataframes back together.
    frames = [slo1_dataframe, slo2_dataframe, slo3_dataframe, slo_dataframe_single_classification]
    slo_dataframe_combined = pd.concat(frames, ignore_index=True)

    # Note: Doing this as context-sensitive menu stopped displaying all useable function calls after concat.
    slo_dataframe_combined = pd.DataFrame(slo_dataframe_combined)

    if debug:
        # Check that we have recombined the dataframes properly.
        log.debug("Recombined individual dataframes for the dataframe representing Tweets with only a single SLO TBL "
                  " topic classification\nand for the dataframes representing Tweets with multiple SLO TBL topic"
                  "classifications:")
        # Iterate through each row and check that each example only has one SLO TBL Classification left.
        for index in slo_dataframe_combined.index:
            log.debug(slo_dataframe_combined['Tweet'][index] + '\tSLO: ' + str(slo_dataframe_combined['SLO'][index]))
        log.debug('\n')
        log.debug('Shape of recombined dataframes: ' + str(slo_dataframe_combined.shape))
        log.debug('\n')

    #######################################################

    # Drop all rows with only NaN in all columns.
    slo_dataframe_combined = slo_dataframe_combined.dropna()

    if debug:
        # Check that we have dropped all NaN's properly.
        log.debug("Recombined dataframes - NaN examples removed:")
        # Iterate through each row and check that we no longer have examples with NaN values.
        for index in slo_dataframe_combined.index:
            log.debug(slo_dataframe_combined['Tweet'][index] + '\tSLO: ' + str(slo_dataframe_combined['SLO'][index]))
        log.debug('\n')
        log.debug('Shape of recombined dataframes without NaN examples: ' + str(slo_dataframe_combined.shape))
        log.debug('\n')

    #######################################################

    # Drop duplicate examples with the same SLO TBL topic classification class.
    slo_dataframe_tbl_duplicates_dropped = slo_dataframe_combined.drop_duplicates(subset=['Tweet', 'SLO'], keep=False)

    if debug:
        # Check that we have dropped all duplicate labels properly.
        log.debug("Duplicate examples with duplicate SLO TBL topic classifications removed:")
        # Iterate through each row and check that we no longer have duplicate examples with the same labels.
        for index in slo_dataframe_tbl_duplicates_dropped.index:
            log.debug(slo_dataframe_tbl_duplicates_dropped['Tweet'][index] + '\tSLO: '
                      + str(slo_dataframe_tbl_duplicates_dropped['SLO'][index]))
        log.debug('\n')
        log.debug(
            'Shape of dataframes without duplicate TBL labels: ' + str(slo_dataframe_tbl_duplicates_dropped.shape))
        log.debug('\n')

    #######################################################

    # Assign new dataframe to contents of old.
    slo_df_tokenized = slo_dataframe_tbl_duplicates_dropped

    # Down-case all text.
    slo_df_tokenized['Tweet'] = slo_df_tokenized['Tweet'].str.lower()

    # Pre-process each tweet individually.
    for index in slo_df_tokenized.index:
        slo_df_tokenized['Tweet'][index] = preprocess_tweet_text(slo_df_tokenized['Tweet'][index])

    # Reindex everything.
    slo_df_tokenized.index = pd.RangeIndex(len(slo_df_tokenized.index))
    # slo_df_tokenized.index = range(len(slo_df_tokenized.index))

    # Save to CSV file.
    slo_df_tokenized.to_csv("preprocessed-datasets/tbl_training_set_PROCESSED.csv", sep=',',
                            encoding='utf-8', index=False, header=['Tweet', 'SLO'])

    # return slo_df_tokenized


################################################################################################################

def tweet_dataset_preprocessor_2():
    """
      Function pre-processes tbl_kvlinden.csv in preparation for machine learning input feature creation.

    :return: Nothing. Saves to CSV file.
    """

    # Import the dataset.
    slo_dataset = \
        pd.read_csv("tbl-datasets/tbl_kvlinden.csv", sep=",")

    # Shuffle the data randomly.
    slo_dataset = slo_dataset.reindex(
        pd.np.random.permutation(slo_dataset.index))

    # Rename columns to something that makes sense.
    column_names = ['Tweet', 'SLO1', 'SLO2']

    # Generate a Pandas dataframe.
    slo_dataframe = pd.DataFrame(slo_dataset)

    if debug:
        # Print shape and column names.
        log.debug("The shape of our SLO dataframe:")
        log.debug(slo_dataframe.shape)
        log.debug("\n")
        log.debug("The columns of our SLO dataframe:")
        log.debug(slo_dataframe.head)
        log.debug("\n")
        log.debug("The 2nd column of our SLO dataframe:")

    # Assign column names.
    slo_dataframe.columns = column_names

    if debug:
        log.debug("The Tweets column:")
        log.debug(slo_dataframe['Tweet'])
        log.debug("\n")
        log.debug("The SLO column:")
        log.debug(slo_dataframe['SLO1'])
        log.debug("\n")
        log.debug("The 2nd SLO column:")
        log.debug(slo_dataframe['SLO2'])
        log.debug("\n")

    #######################################################

    # Restrict to just SLO1 column by dropping SLO2 column.
    slo_dataframe_column1 = slo_dataframe.drop('SLO2', axis=1)

    if debug:
        log.debug("The shape of dataframe with only slo column1:")
        log.debug(slo_dataframe_column1.shape)
        log.debug("\n")
        log.debug("The contents of the dataframe with only slo column1:")
        log.debug(slo_dataframe_column1.sample())
        log.debug("\n")

    #######################################################

    # Drop any row with "NaN" columns. (isolates examples with multiple TBL classification labels)
    slo_dataframe_column2 = slo_dataframe.dropna()

    if debug:
        log.debug("The contents of the dataframe with only examples containing multiple classifications")
        for index in slo_dataframe_column2.index:
            log.debug(slo_dataframe_column2['Tweet'][index] + '\tSLO1: '
                      + str(slo_dataframe_column2['SLO1'][index])
                      + '\tSLO2: ' + str(slo_dataframe_column2['SLO2'][index]))
        log.debug("\n")

    # Drop SLO1 column to restrict to just 2nd classification label in SLO2 column.
    slo_dataframe_column2 = slo_dataframe_column2.drop('SLO1', axis=1)

    #######################################################

    # Rename columns for concatenation back into a single dataframe.
    column_names = ["Tweet", "SLO"]
    slo_dataframe_column1.columns = column_names
    slo_dataframe_column2.columns = column_names

    if debug:
        log.debug("Check that we have renamed columns properly:")
        log.debug(slo_dataframe_column1.head())
        log.debug(slo_dataframe_column2.head())
        log.debug("\n")

    # Concatenate the individual dataframes back together.
    frames = [slo_dataframe_column1, slo_dataframe_column2]
    slo_dataframe_combined = pd.concat(frames, ignore_index=True)

    if debug:
        log.debug("Check that we have concatenated properly:")
        log.debug(slo_dataframe_combined.shape)
        log.debug("\n")
        log.debug(slo_dataframe_combined.tail())
        log.debug("\n")

    #######################################################

    # Down-case all text.
    slo_dataframe_combined['Tweet'] = slo_dataframe_combined['Tweet'].str.lower()

    # Pre-process each tweet individually.
    for index in slo_dataframe_combined.index:
        slo_dataframe_combined['Tweet'][index] = preprocess_tweet_text(slo_dataframe_combined['Tweet'][index])

    # Reindex everything.
    slo_dataframe_combined.index = pd.RangeIndex(len(slo_dataframe_combined.index))
    # slo_dataframe_combined.index = range(len(slo_dataframe_combined.index))

    if debug:
        log.debug("Check that we have pre-processed properly:")
        for index in slo_dataframe_combined.index:
            log.debug(slo_dataframe_combined['Tweet'][index] + '\tSLO: '
                      + str(slo_dataframe_combined['SLO'][index]))
        log.debug("\n")

    # Save to CSV file.
    slo_dataframe_combined.to_csv("preprocessed-datasets/tbl_kvlinden_PROCESSED.csv", sep=',',
                                  encoding='utf-8', index=False, header=['Tweet', 'SLO'])

    # return slo_dataframe_combined


################################################################################################################

def tweet_dataset_preprocessor_3():
    """
    Function pre-processes dataset_20100101-20180510_tok.csv in preparation for machine learning input feature creation.

    Note: We are doing this as our pre-processing for the other datasets we are using is different from the
    pre-processing done on this already tokenized dataset.  Hence, we wish to normalize the difference between them
    as much as possible before using this dataset as our prediction set.

    :return: Nothing. Saves to CSV file.
    """

    # Import the dataset.
    slo_dataset_cmu = \
        pd.read_csv("borg-SLO-classifiers/dataset_20100101-20180510_tok.csv", sep=",")

    # Shuffle the data randomly.
    slo_dataset_cmu = slo_dataset_cmu.reindex(
        pd.np.random.permutation(slo_dataset_cmu.index))

    # Generate a Pandas dataframe.
    slo_dataframe_cmu = pd.DataFrame(slo_dataset_cmu)

    # Print shape and column names.
    log.debug("\n")
    log.debug("The shape of our SLO CMU dataframe:")
    log.debug(slo_dataframe_cmu.shape)
    log.debug("\n")
    log.debug("The columns of our SLO CMU dataframe:")
    log.debug(slo_dataframe_cmu.head)
    log.debug("\n")

    #######################################################

    # Down-case all text.
    slo_dataframe_cmu['tweet_t'] = slo_dataframe_cmu['tweet_t'].str.lower()

    # Pre-process each tweet individually.
    for index in slo_dataframe_cmu.index:
        slo_dataframe_cmu['tweet_t'][index] = preprocess_tweet_text(slo_dataframe_cmu['tweet_t'][index])

    # Reindex everything.
    slo_dataframe_cmu.index = pd.RangeIndex(len(slo_dataframe_cmu.index))
    # slo_dataframe_combined.index = range(len(slo_dataframe_combined.index))

    # Create input features.
    selected_features_cmu = slo_dataframe_cmu['tweet_t']
    processed_features_cmu = selected_features_cmu.copy()

    # Check what we are using for predictions.
    if debug:
        log.debug("The shape of our SLO CMU feature dataframe:")
        log.debug(processed_features_cmu.shape)
        log.debug("\n")
        log.debug("The columns of our SLO CMU feature dataframe:")
        log.debug(processed_features_cmu.head)
        log.debug("\n")

    # Save to CSV file.
    slo_dataframe_cmu.to_csv("preprocessed-datasets/dataset_20100101-20180510_tok_PROCESSED.csv", sep=',',
                             encoding='utf-8', index=False)

    # return processed_features_cmu


################################################################################################################

############################################################################################
"""
Main function.  Execute the program.

Note: Used to individually test that the preprocessors function as intended.
"""

if __name__ == '__main__':

    start_time = time.time()

    """
    Comment or uncomment in order to run the associated tweet preprocessor module.
    """
    # tweet_dataset_preprocessor_1()
    # tweet_dataset_preprocessor_2()
    # tweet_dataset_preprocessor_3()

    end_time = time.time()

    if debug:
        log.debug("\n")
        log.debug("Time taken to run pre-processor function:")
        time_taken = end_time - start_time
        log.debug(time_taken)
        log.debug("\n")

############################################################################################
