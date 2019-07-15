"""
Social License to Operate Research
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 6-25-19
version: 2.0

SLO Twitter Dataset Analysis Utility Functions

###########################################################
Notes:

These are used by and imported into the "slo_twitter_data_analysis.py" file.

###########################################################

Resources Used:

dataset_slo_20100101-20180510.json

"""

################################################################################################################
################################################################################################################

import logging as log
import warnings
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import json

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
"""
This section contains helper functions taken from "SLO analysis.ipynb" by Shuntaro Yada from CSIRO.
"""


def bar_plot(col, **kwargs):
    """
    Helper function to visualize the data.

    :param col: the columns of the graph.
    :param kwargs: variable number of arguments.
    :return: None.
    """
    ax = plt.gca()
    data = kwargs.pop('data')
    height = data[col].value_counts(normalize=True)
    height.sort_index(inplace=True)
    ax.bar(height.index, height)


################################################################################################################

def bar_plot_zipf(col, **kwargs):
    """
    Helper function to visualize the data.

    Based on Zipf's Law. (https://en.wikipedia.org/wiki/Zipf%27s_law)

    :param col: the columns of the graph.
    :param kwargs: variable number of arguments.
    :return: None.
    """
    ax = plt.gca()
    data = kwargs.pop('data')
    height = data[col].value_counts().value_counts(normalize=True)
    ax.bar(height.index, height)


################################################################################################################

def ts_plot(col, **kwargs):
    """
    Helper function to visualize the data.
    Used specifically for Time-Series Statistics.

    :param col: the columns of the graph.
    :param kwargs: variable number of arguments.
    :return: None.
    """
    ax = plt.gca()
    data = kwargs.pop('data')
    ts = pd.to_datetime(data[col]).value_counts().resample('1D').sum()
    ax.plot(ts)


################################################################################################################

def ts_plot_2(col, **kwargs):
    """
    Helper function to visualize the data.
    Used specifically for Time-Series Statistics.

    :param col: the columns of the graph.
    :param kwargs: variable number of arguments.
    :return: None.
    """
    ax = plt.gca()
    data = kwargs.pop('data')

    ts_rt = pd.to_datetime(data.query("retweeted_derived == '1'")[col]).value_counts().resample('1D').sum()
    ts = pd.to_datetime(data.query("retweeted_derived == '0'")[col]).value_counts().resample('1D').sum()

    ax.plot(ts)
    ax.plot(ts_rt)


################################################################################################################

def ts_plot_2_fixed(col, **kwargs):
    """
    Helper function to visualize the data.
    Used specifically for Time-Series Statistics.

    :param col: the columns of the graph.
    :param kwargs: variable number of arguments.
    :return: None.
    """
    ax = plt.gca()
    data = kwargs.pop('data')
    data_df = pd.DataFrame(data)

    yes_retweet = data_df.loc[data_df["retweeted_derived"] == True]
    no_retweet = data_df.loc[data_df["retweeted_derived"] == False]

    ts_rt = pd.to_datetime(yes_retweet[col]).value_counts().resample('1D').sum()
    ts = pd.to_datetime(no_retweet[col]).value_counts().resample('1D').sum()

    ax.plot(ts)
    ax.plot(ts_rt)


################################################################################################################

def relhist_proc(col, **kwargs):
    """
    Helper function to visualize the data.  Constructs a relative frequency histogram.

    :param col: the columns of the graph.
    :param kwargs: variable number of arguments.
    :return: None.
    """
    ax = plt.gca()
    data = kwargs.pop('data')
    proc = kwargs.pop('proc')
    processed = proc(data[col])
    # relative frequency histogram
    # https://stackoverflow.com/questions/9767241/setting-a-relative-frequency-in-a-matplotlib-histogram
    ax.hist(processed, weights=np.ones_like(processed) / processed.size, **kwargs)


################################################################################################################

def char_len(tweets):
    """
    Determine the length of the Tweet text.

    :param tweets: the Tweet text.
    :return: the length of the Tweet.
    """
    return tweets.str.len()


################################################################################################################
################################################################################################################
"""
This section contains the utility functions we created.
"""


def import_dataset(input_file_path, file_type, show_df_info):
    """
    This function imports a JSON or CSV dataset and puts it into a Pandas Dataframe while providing basic information
    on the contents of the data in the frame.

    Note: Does NOT import in chunks.  Assumes file will fit in system RAM.

    :return: the Pandas Dataframe containing the dataset.
    """
    if file_type == "csv":
        # Read in the CSV file.
        tweet_dataset = pd.read_csv(f"{input_file_path}", sep=",", encoding="utf-8")
    elif file_type == "json":
        # Read in the JSON file.
        tweet_dataset = pd.read_json(f"{input_file_path}", orient='records', lines=True)
    else:
        print("Invalid file type - aborting operation")
        return

    # Generate a Pandas dataframe.
    dataframe = pd.DataFrame(tweet_dataset)

    if show_df_info:
        # Print shape and column names.
        log.info(f"\nThe shape of our dataframe storing the contents of the {file_type} Tweet data is:\n")
        log.info(dataframe.shape)
        log.info(f"\nThe columns of our dataframe storing the contents of the {file_type} Tweet data is:\n")
        log.info(dataframe.columns)
        log.info(f"\nThe first row from the dataframe storing the contents of the {file_type} Tweet data is:\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None, 'display.width', None, 'display.max_colwidth', 1000):
            log.info(f"\n{dataframe.iloc[0]}\n")
    return dataframe


################################################################################################################


def export_to_csv_json(tweet_dataframe, attribute_list, output_file_path, export_mode, file_type):
    """
    This function exports to a CSV file.

    Note: Use "[]" or empty List if you wish to save out entire Dataframe without specifying columns.
    Note: export_mode is only for CSV.  use "w" = write or "a" = append.

    :param tweet_dataframe:  Tweet dataframe to export.
    :param attribute_list: names of the attributes (columns) to export to the CSV dataset file.
    :param output_file_path: absolute file path of location to write to (including file-name excluding file-extension).
    :param export_mode: "w" for write (overwrites file) or "a" for append (appends to file)
    :param file_type: export as a "csv" or "json" file.
    :return: None. Saves to file.
    """
    if len(attribute_list) > 0 and file_type == "json":
        tweet_dataframe[attribute_list].to_json(f"{output_file_path}.json", orient='records', lines=True)

    elif len(attribute_list) > 0 and file_type == "csv":
        tweet_dataframe[attribute_list].to_csv(f"{output_file_path}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC,
                                               mode=export_mode, header=True, encoding="utf-8")

    elif len(attribute_list) == 0 and file_type == "json":
        tweet_dataframe.to_json(f"{output_file_path}.json", orient='records', lines=True)

    elif len(attribute_list) == 0 and file_type == "csv":
        tweet_dataframe.to_csv(f"{output_file_path}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC,
                               mode=export_mode, header=True, encoding="utf-8")
    else:
        print("Invalid export mode or file type entered!")


################################################################################################################

def call_data_analysis_function_on_json_file_chunks(input_file_path, function_name):
    """
    This function reads the raw JSON Tweet dataset in chunk-by-chunk and calls the user-defined data analysis
    function that is specified as a parameter.

    :param input_file_path: absolute file path of the input raw JSON file.
    :param function_name: name of the data analysis function to call.
    :return: None.
    """
    start_time = time.time()

    # Define size of chunks to read in.
    chunksize = 100000

    # Read in the JSON file.
    twitter_data = pd.read_json(f"{input_file_path}",
                                orient='records',
                                lines=True,
                                chunksize=chunksize)

    # Create a empty Pandas dataframe.
    json_dataframe = pd.DataFrame()

    chunk_number = 0

    # Loop through chunk-by-chunk and call the data analysis function on each chunk.
    for data in twitter_data:
        json_dataframe = json_dataframe.append(data, ignore_index=True)
        chunk_number += 1

        if chunk_number == 1 and function_name == "none":
            # Print shape and column names.
            log.info(
                f"\nThe shape of the dataframe storing the contents of the raw JSON Tweet file chunk "
                f"{chunk_number} is:\n")
            log.info(json_dataframe.shape)
            log.info(
                f"\nThe columns of the dataframe storing the contents of the raw JSON Tweet file chunk "
                f"{chunk_number} is:\n")
            log.info(json_dataframe.columns)
            log.info(
                f"\nThe first row from the dataframe storing the contents of the raw JSON Tweet file chunk "
                f"{chunk_number} is:\n")
            with pd.option_context('display.max_rows', None, 'display.max_columns',
                                   None, 'display.width', None, 'display.max_colwidth', 1000):
                log.info(f"\n{json_dataframe.iloc[0]}")
            time.sleep(2)

        if function_name != "none":
            # Call the data analysis functions.
            function_name(json_dataframe, chunk_number)
        else:
            return
        # Clear the contents of the dataframe.
        json_dataframe = pd.DataFrame()

        # For debug purposes.
        # break

    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60.0
    time.sleep(3)
    log.info(f"The time taken to read in the JSON file by Chunks is {time_elapsed} minutes")
    log.info(f"The number of chunks is {chunk_number} based on chunk size of {chunksize}")
    log.info('\n')


################################################################################################################

def generalized_data_chunking_file_export_function(input_file_path, output_file_path, file_type):
    """
    This function reads in raw JSON or CSV files as chunks and exports each individual chunk to a CSV or JSON file.
    Please use absolute file path strings.
    Please use "csv" or "json" for the file type to save as and import as.

    :param file_type: type of file to save as or import as.
    :param input_file_path: absolute file path of the input file.
    :param output_file_path: absolute file path of the output file.

    :return: None.
    """
    start_time = time.time()

    # Define size of chunks to read in.
    chunksize = 100000

    if file_type == "csv":
        # Read in the CSV file.
        twitter_data = \
            pd.read_csv(f"{input_file_path}", sep=",",
                        chunksize=chunksize)
    elif file_type == "json":
        # Read in the JSON file.
        twitter_data = pd.read_json(f"{input_file_path}",
                                    orient='records',
                                    lines=True,
                                    chunksize=chunksize)
    else:
        print("Invalid file type - aborting operation")
        return

    # Create a empty Pandas dataframe.
    json_dataframe = pd.DataFrame()

    # Read in data by chunk and export to file.
    chunk_number = 0
    for data in twitter_data:
        json_dataframe = json_dataframe.append(data, ignore_index=True)
        chunk_number += 1

        # Note: Absolute file paths are required.  Relative do not work.
        csv_output_file_path = f"{output_file_path}raw-twitter-dataset-chunk-{chunk_number}.csv"
        json_output_file_path = f"{output_file_path}raw-twitter-dataset-chunk-{chunk_number}.json"

        if file_type == "csv":
            # Export to CSV file.
            json_dataframe.to_csv(csv_output_file_path, sep=',',
                                  encoding='utf-8', index=False, header=True)
        elif file_type == "json":
            # Export JSON file.
            json_dataframe.to_json(json_output_file_path, orient='records', lines=True)
        else:
            print(f"Invalid file type entered - aborting operation")
            return

        # Clear the contents of the dataframe.
        json_dataframe = pd.DataFrame()

    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60.0

    log.info(f"The time taken to read in the JSON file by Chunks is {time_elapsed} minutes")
    log.info(f"The number of chunks is {chunk_number} based on chunk size of {chunksize}")
    log.info('\n')


################################################################################################################

def tweet_single_company_name_or_multiple_company_designation(tweet_dataframe):
    """
    This function adds a attribute to the dataset that identifies a Tweet as being associated with a single company
    by that company's name or if associated with multiple companies, by the designation of "multiple".

    Note: This utility function is intended to also be implemented in "dataset_process_adapted.py" to add a new column
    to the CSV dataset produced from the raw JSON datset.

    :param tweet_dataframe: Tweet dataframe.
    :return: None.
    """

    def compute_company_designation(row):
        """
         This function adds a attribute to the dataset that identifies a Tweet as being associated with a single
         company by that company's name or if associated with multiple companies, by the designation of "multiple".

        :param row: example in the dataset we are operating on.
        :return:  the modified example.
        """

        if row["multiple_companies_derived_count"] > 1:
            row["company_derived_designation"] = "multiple"
        else:
            row["company_derived_designation"] = row["company_derived"]

        return row["company_derived_designation"]

    dataframe = pd.DataFrame(tweet_dataframe)
    # Note: Ensure axis=1 so function applies to entire row rather than per column by default. (axis = 0 = column)
    dataframe["company_derived_designation"] = dataframe.apply(compute_company_designation, axis=1)

    export_to_csv_json(
        dataframe, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/selected-attributes-final", "w", "csv")


################################################################################################################

def export_multi_company_tweets(tweet_dataframe):
    """
    This function exports to a CSV dataset file only those Tweets that are associated with multiple companies.

    :param tweet_dataframe: Tweet dataframe.
    :return: None.
    """
    dataframe = pd.DataFrame(tweet_dataframe)
    multi_company_only_df = dataframe.loc[dataframe['company_derived_designation'] == "multiple"]
    export_to_csv_json(
        multi_company_only_df, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/multi-company-tweets", "w", "csv")


################################################################################################################

def export_no_company_tweets(tweet_dataframe):
    """
    This function exports to a CSV dataset file only those Tweets that are associated with no companies.

    :param tweet_dataframe: Tweet dataframe.
    :return: None.
    """
    dataframe = pd.DataFrame(tweet_dataframe)
    no_company_only_df = dataframe.loc[(dataframe['company_derived_designation'].isnull())]
    export_to_csv_json(
        no_company_only_df, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/no-company-tweets", "w", "csv")


################################################################################################################

def export_non_english_tweets(tweet_dataframe):
    """
    This function exports to a CSV dataset file only those Tweets that are designated as non-English by both
    spacy-langdetect and the Twitter API.

    :param tweet_dataframe: Tweet dataframe.
    :return: None.
    """
    dataframe = pd.DataFrame(tweet_dataframe)
    non_english_spacy_and_twitter = tweet_dataframe.loc[(tweet_dataframe["spaCy_language_detect_all_tweets"] != "en") &
                                                        (tweet_dataframe["tweet_lang"] != "en")]
    export_to_csv_json(
        non_english_spacy_and_twitter, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/non-english-tweets", "w", "csv")


################################################################################################################

def extract_tweets_over_specified_character_length(tweet_dataframe, character_length):
    """
    This function extracts all tweets over the specified character length and exports it to a separate CSV file.

    :param tweet_dataframe: Tweet dataframe.
    :param character_length: Tweets over this length should be extracted.
    :return: None.
    """
    long_tweets = tweet_dataframe.loc[tweet_dataframe["tweet_text_length_derived"] > character_length]
    export_to_csv_json(
        long_tweets, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/tweets-over-140-characters", "w", "csv")


################################################################################################################

def compute_user_description_text_length(tweet_dataframe):
    """
    This function adds a attribute to the dataset that records the length of the user description text.

    Note: This utility function is intended to also be implemented in "dataset_process_adapted.py" to add a new column
    to the CSV dataset produced from the raw JSON datset.

    Resources:
    https://stackoverflow.com/questions/26614465/python-pandas-apply-function-if-a-column-value-is-not-null

    :param tweet_dataframe: Tweet dataframe.
    :return: None.
    FIXME - currently only properly calculates length for CSV files, NOT JSON files.
    """

    def text_length(row):
        """
        Helper function that determines the length of the user description text.
        :param row: example in the dataset we are operating on.
        :return:  the modified example.
        """
        row["user_description_text_length"] = len(row['user_description'])
        return row["user_description_text_length"]

    dataframe = pd.DataFrame(tweet_dataframe)
    # Note: Ensure axis=1 so function applies to entire row rather than per column by default. (axis = 0 = column)
    dataframe["user_description_text_length"] = dataframe.apply(
        lambda x: text_length(x) if (pd.notnull(x["user_description"])) else 0, axis=1)

    export_to_csv_json(
        dataframe, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/selected-attributes-final--subset-test",
        "w", "csv")


################################################################################################################

def extract_single_multi_json_attributes(input_file_path, output_file_path, attributes_list, output_file_type):
    """
    Function to flatten a JSON file structure and extract specified attribute(s) to CSV or JSON file.

    :param input_file_path: absolute file path of the input data file (including file extension ".file_type"
    :param output_file_path: absolute file path of the output data file save location
            (excluding file extension ".file_type")
    :param attributes_list: attribute(s) to extract as a List of attributes.
    :param output_file_type: file type to output as.
    :return: None.  Saves to file.
    """
    row_counter = 0

    def extract_attributes(row):
        """
        Helper function to extract attributes from each example in the dataframe.
        Not necessary to call "df.apply(func,axis=1)" just to extract non-nested list of attribute(s),
        but keeping this here in case we want to modify something while extracting attribute(s) in the future.

        :param row: current example (row) passed in.
        :return: nested attributes as individual columns added to the current example (row).
        """
        nonlocal row_counter
        row_counter += 1
        log.debug(f"\nCurrent row {row_counter}:\n")
        log.debug(f"{row}\n")
        return row[attributes_list]

    count_records = 0
    include_header = True
    # Read Json in chunks to avoid RAM issues.
    for df_chunk in pd.read_json(input_file_path, orient='records', lines=True, chunksize=1000):
        # Extract all nested attributes.
        df_chunk[attributes_list] = df_chunk.apply(extract_attributes, axis=1)

        # Write each chunk to the combined dataset file.
        df_chunk[attributes_list].to_csv(f"{output_file_path}-temp-please-delete.csv", index=False,
                                         quoting=csv.QUOTE_NONNUMERIC,
                                         mode='a', header=include_header, encoding="utf-8")
        # Print a progress message.
        count_records += df_chunk.shape[0]
        print(f'\t\tprocessed {count_records} records...')
        # Only include the header once, at the top of the file.
        include_header = False
        # break

    # Remove empty rows in CSV indicating "retweeted_status" was null for that row in the Twitter dataset.
    df_full = pd.read_csv(f"{output_file_path}-temp-please-delete.csv", sep=',')
    df_full.dropna(how="all", inplace=True)

    # Save file as type specified by user in function call.
    if output_file_type == "csv":
        df_full.to_csv(f"{output_file_path}.csv", mode='w', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC,
                       encoding='utf-8')
    elif output_file_type == "json":
        df_full.to_json(f"{output_file_path}.json", orient='records', lines=True)
    else:
        print("Invalid file type! - 'json' or 'csv' are accepted.")

    # Delete the temp file to avoid appending to output during next execution, if in same location.
    import os
    if os.path.exists(f"{output_file_path}-temp-please-delete.csv"):
        os.remove(f"{output_file_path}-temp-please-delete.csv")
    else:
        print(f"Cannot find {output_file_path}-temp-please-delete.csv - please delete manually, if necessary.")


################################################################################################################

def flatten_extract_nested_json_attributes(input_file_path, output_file_path, top_level_attribute,
                                           nested_attributes_list, output_file_type):
    """
    Function to flatten nested fields in a JSON file structure and extract to CSV OR JSON file.

    :param input_file_path: absolute file path of the input data file (including file extension ".file_type"
    :param output_file_path: absolute file path of the output data file save location
            (excluding file extension ".file_type")
    :param top_level_attribute: outer attribute that encapsulates the nested attributes.
    :param nested_attributes_list: nested attributes to extract as a List of attributes.
    :param output_file_type: file type to output as.
    :return: None.  Saves to file.
    """
    series_counter = 0

    def flatten_json(row):
        """
        Helper function to extract attributes from each example in the dataframe.

        :param row: current example (row) passed in.
        :return: nested attributes as individual columns added to the current example (row).
        """
        nonlocal series_counter
        series_counter += 1
        if not pd.isnull(row[top_level_attribute]):
            series = pd.read_json(json.dumps(row[top_level_attribute]), typ='series')
            log.debug(f"\nCurrent series {series_counter}:\n")
            log.debug(f"{series}\n")
            return series[nested_attributes_list]
        row[nested_attributes_list] = np.NaN
        return row[nested_attributes_list]

    count_examples = 0
    include_header = True
    # Read Json in chunks to avoid RAM issues.
    for df_chunk in pd.read_json(input_file_path, orient='records', lines=True, chunksize=1000):
        # Extract all nested attributes.
        df_chunk[nested_attributes_list] = df_chunk.apply(flatten_json, axis=1)

        # Write each chunk to the combined dataset file.
        df_chunk[nested_attributes_list].to_csv(f"{output_file_path}-temp-please-delete.csv", index=False,
                                                quoting=csv.QUOTE_NONNUMERIC,
                                                mode='a', header=include_header, encoding="utf-8")
        # Print a progress message.
        count_examples += df_chunk.shape[0]
        print(f'\t\tprocessed {count_examples} records...')
        # Only include the header once, at the top of the file.
        include_header = False
        # break

    # Remove empty rows in CSV indicating "retweeted_status" was null for that row in the Twitter dataset.
    df_full = pd.read_csv(f"{output_file_path}-temp-please-delete.csv", sep=',')
    df_full.dropna(how="all", inplace=True)

    # Save file as type specified by user in function call.
    if output_file_type == "csv":
        df_full.to_csv(f"{output_file_path}.csv", mode='w', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC,
                       encoding='utf-8')
    elif output_file_type == "json":
        df_full.to_json(f"{output_file_path}.json", orient='records', lines=True)
    else:
        print("Invalid file type! - 'json' or 'csv' are accepted.")

    # Delete the temp file to avoid appending to output during next execution, if in same location.
    import os
    if os.path.exists(f"{output_file_path}-temp-please-delete.csv"):
        os.remove(f"{output_file_path}-temp-please-delete.csv")
    else:
        print(f"Cannot find {output_file_path}-temp-please-delete.csv - please delete manually, if necessary.")


################################################################################################################


def spacy_language_detection(tweet_dataframe):
    """
    This function adds a attribute to the dataset that records what language the Tweet is in.
    We use the "spaCy" natural language processing library, "langdetect" language detection library, and the
    "spacy-langdetect" library built on top of them to identify Tweets as English or non-English.

    Note: This utility function is intended to also be implemented in "dataset_process_adapted.py" to add a new column
    to the CSV dataset produced from the raw JSON datset.

    :param tweet_dataframe: Tweet dataframe.
    :return: None. Saves to file.
    """

    # noinspection PyProtectedMember
    def what_language(row):
        """
         This helper function executes spaCy N.L.P. library and "spacy-langdetect"
         to determine the language of the Tweet.

        :param row: example in the dataset we are operating on.
        :return:  the modified example with additional column specifying its language.
        """
        import spacy
        from spacy_langdetect import LanguageDetector
        nlp = spacy.load("en")
        nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
        document = nlp(row["text_derived"])
        # document level language detection. Think of it like average language of document!
        text_language = document._.language
        row["spaCy_language_detect"] = str(text_language["language"])
        return row["spaCy_language_detect"]

    dataframe = pd.DataFrame(tweet_dataframe)
    # Ensure axis=1 so function applies to entire row rather than per column by default. (axis=0 is column)
    dataframe["spaCy_language_detect"] = dataframe.apply(
        lambda x: what_language(x) if (pd.notnull(x["text_derived"])) else "none", axis=1)

    export_to_csv_json(
        dataframe, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/spacy-lang-detect-test", "w", "csv")


################################################################################################################

def indicate_is_retweet(tweet_dataframe):
    """
    This function adds a attribute (column) to the dataset that indicates that the Tweet is a re-Tweet.
    :param tweet_dataframe: Tweet dataframe.
    :return: None. Saves to file.
    """

    def is_a_retweet(row):
        """
        This helper function determines whether a Tweet in our dataset is a re-tweet.
        :param row: example in the dataset we are operating on.
        :return:  the modified example with additional column specifying if it's a re-tweet.
        TODO - figure out a way to determine if retweet based on presence of non-null value for retweeted_status
        """
        temp_df = pd.DataFrame(row)
        # temp_series = pd.Series(row)
        # print(temp_df)
        # print(temp_series)
        if "retweeted_status" in temp_df.columns:
            row["is_a_retweet"] = True
            return row["is_a_retweet"]
        row["is_a_retweet"] = False
        return row["is_a_retweet"]

    dataframe = pd.DataFrame(tweet_dataframe)
    dataframe["is_a_retweet?"] = dataframe.apply(is_a_retweet, axis=1)

    export_to_csv_json(
        dataframe, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/selected-attributes-final-is-retweet",
        "w", "csv")


################################################################################################################

def find_mixed_data_types_in_dataset_rows(tweet_dataframe):
    """
    This function finds mixed data types that should not exist for the specified column(s) in the dataset.

    Note: Takes way too long to run on a large file.

    :param tweet_dataframe: Tweet dataframe.
    :return: None. Saves to file.
    """

    # noinspection PyUnresolvedReferences
    weird = (tweet_dataframe.applymap(type) != tweet_dataframe.iloc[1].apply(type)).any(axis=1)

    print("These rows contain data type(s) that is different from the rest of the rows in the column(s):")
    print(tweet_dataframe[weird])


################################################################################################################

counter = 0


def determine_multiple_companies_count_fixed(tweet_dataframe):
    """
    This function tests that determining the # of companies a Tweet is associated with is fixed.

    :param tweet_dataframe: Tweet dataframe.
    :return: None. Saves to file.
    """

    def compute_multiple_companies_count(row):
        """
        Function determines the number of companies a Tweet is associated with.
        Note: we convert derived_series to a series to avoid Pandas warning.
        :param row: example in the dataset we are operating on.
        :return:  the modified example.
        """
        global counter
        counter += 1
        derived_series = pd.read_json(json.dumps(row['company_derived']), typ='series')
        derived_series = pd.Series(derived_series)
        derived_string = derived_series.to_string()
        print(f"counter value is: {counter}")
        print(f"Length of derived string: {len(derived_string)}")
        print(f"The derived string: {derived_string}")
        if derived_string.count('|') > 0:
            row["multiple_companies_derived_count"] = derived_string.count('|') + 1
        elif "none" in derived_string:
            row["multiple_companies_derived_count"] = 0
        else:
            row["multiple_companies_derived_count"] = 1
        return row["multiple_companies_derived_count"]

    dataframe = pd.DataFrame(tweet_dataframe)
    dataframe["multiple_companies_derived_count"] = dataframe.apply(compute_multiple_companies_count, axis=1)

    export_to_csv_json(
        dataframe, [],
        "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/compute-multiple-companies-count-debug",
        "w", "csv")


################################################################################################################

# tweet_csv_dataframe = import_dataset(
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "twitter-dataset-6-27-19.csv",
#     "csv", False)

# tweet_csv_dataframe_2 = import_dataset(
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/"
#     "debug.json",
#     "json", False)

# export_to_csv_json(
#     tweet_csv_dataframe, [],
#     "D:/Dropbox/summer-research-2019/jupyter-notebooks/attribute-datasets/debug", "w", "json")

# determine_multiple_companies_count_fixed(tweet_csv_dataframe_2)

# export_no_company_tweets(tweet_csv_dataframe)
# export_non_english_tweets(tweet_csv_dataframe)
