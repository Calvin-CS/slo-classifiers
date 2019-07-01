"""
This Python file contains modules that processes a raw JSON Twitter data file into a formatted CSV file containing
various derived attributes and native attributes.

Original Author: Professor Keith VanderLinden
Modified by: Joseph Jinn
Modified for: SLO Twitter Data Analysis for Topic Modeling Research.
Date: June 19, 2019
Version: 2.0

Notes:

Ensure there is at least ONE existing non-null value for any attribute you wish to rename in the current data chunk,
otherwise you will encounter a error along the lines of: KeyError: 'quoted_status_id'

# TODO - parallelize the data chunks to reduce computational time, if possible.
# TODO - clean up this file now that we have added Linux compatability support (or split into a windows version)
"""

#########################################################################################################
#########################################################################################################

import os
import csv
import json
import re
import sys
import time
from os import path
import numpy as np
import pandas as pd
import logging as log
from pathlib import Path
# from fire import Fire
import spacy
from spacy_langdetect import LanguageDetector
from settings import PTN_rt, PTN_companies

# import dask.dataframe as dask_df

log.basicConfig(level=log.INFO)
# Count irrelevant tweets.
unknown_company_count_global = 0
non_english_count_global = 0

# from polyglot.text import Text, Word

#########################################################################################################
"""
These Lists hold the attributes/fields/columns we wish to extract/derive from the JSON and export to the CSV/JSON.
"""

# Original Tweet object attribute names present in raw JSON file.
original_tweet_object_field_names = [
    'created_at', 'id', 'full_text', 'in_reply_to_status_id', 'in_reply_to_user_id',
    'in_reply_to_screen_name', 'retweet_count', 'favorite_count', 'lang']

# Names to rename main Tweet object attributes.
tweet_object_fields = [
    'tweet_created_at', 'tweet_id', 'tweet_full_text', 'tweet_in_reply_to_status_id',
    'tweet_in_reply_to_user_id', 'tweet_in_reply_to_screen_name', 'tweet_retweet_count',
    'tweet_favorite_count', 'tweet_lang']

# Names to give "user" object attributes.
user_object_fields = [
    'user_id', 'user_name', 'user_screen_name', 'user_location', 'user_description',
    'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count',
    'user_statuses_count', 'user_created_at', 'user_time_zone', 'user_lang']

# Names to give "entities" object attributes.
entities_object_fields = [
    "tweet_entities_expanded_urls", "tweet_entities_hashtags", "tweet_entities_user_mentions_id",
    "tweet_entities_user_mentions_name", "tweet_entities_user_mentions_screen_name",
    "tweet_entities_symbols"]

# Names to give "retweeted_status" object attributes.
retweeted_status_object_fields = [
    'retweeted_status_created_at', 'retweeted_status_id', 'retweeted_status_full_text',
    'retweeted_status_in_reply_to_status_id', 'retweeted_status_in_reply_to_user_id',
    'retweeted_status_in_reply_to_screen_name', 'retweeted_status_retweet_count',
    'retweeted_status_favorite_count', 'retweeted_status_lang',
    'retweeted_status_entities',
    'retweeted_status_user', 'retweeted_status_coordinates', 'retweeted_status_place']

# Names to give "user" object attributes.
retweeted_status_user_object_fields = [
    'retweeted_status_user_id', 'retweeted_status_user_name', 'retweeted_status_user_screen_name',
    'retweeted_status_user_location', 'retweeted_status_user_description', 'retweeted_status_user_followers_count',
    'retweeted_status_user_friends_count', 'retweeted_status_user_listed_count',
    'retweeted_status_user_favourites_count', 'retweeted_status_user_statuses_count',
    'retweeted_status_user_created_at', 'retweeted_status_user_time_zone', 'retweeted_status_user_lang']


#########################################################################################################
#########################################################################################################

def create_dataset(json_data_filepath, dataset_filepath, drop_irrelevant_tweets):
    """
    Function creates a Twitter CSV dataset file from raw JSON dataset file.

    Note: The order by which we use "df.apply(func) IS IMPORTANT as there is some dependencies between derived fields.

    :param json_data_filepath: absolute filepath of the raw JSON file (include the file type extension)
    :param dataset_filepath:  absolute filepath of the location to save to (exclude file type extension)
    :param drop_irrelevant_tweets: remove Tweets declared irrelevant from the dataframe/dataset.
    :return: None.  Exports to CSV/JSON file.
    """
    # Stupidity check.
    check_for_preexisting_output_file(dataset_filepath)

    global unknown_company_count_global, non_english_count_global
    # log.info(f'\tloading raw tweets from {json_data_filepath}')
    log.info("\tloading raw tweets from " + json_data_filepath)

    # Load/save the file in chunks.
    count = 0
    include_header = True

    # # Parallelize the chunks.
    # for df_chunk in dask_df.read_json(json_data_filepath, orient='records', lines=True, blocksize=10000,
    #                                   encoding="ISO-8859-1"):
    #     print()

    for df_chunk in pd.read_json(json_data_filepath, orient='records', lines=True, chunksize=10000):

        # Modify these to determine what to export to CSV.
        required_fields = ['retweeted_derived', 'company_derived', 'text_derived',  # "tweet_quoted_status_id",
                           'tweet_url_link_derived', 'multiple_companies_derived_count', "company_derived_designation",
                           'tweet_text_length_derived', "spaCy_language_detect_all_tweets",
                           "user_description_text_length",  # "polyglot_lang_detect_all_tweets"
                           ] + tweet_object_fields + user_object_fields + entities_object_fields

        extra_fields = ["tweet_id"] + retweeted_status_object_fields

        # Rename main Tweet object fields.
        df_chunk[tweet_object_fields] = df_chunk[original_tweet_object_field_names]

        # FIXME - KeyError: ('quoted_status_id', 'occurred at index 0') - debug the issue.
        # df_chunk["tweet_quoted_status_id"] = df_chunk.apply(rename_column, axis=1)

        # Extract Tweet "user" object fields.
        df_chunk[user_object_fields] = df_chunk.apply(compute_user_series, axis=1)

        # Determine the user profile description text length.
        df_chunk["user_description_text_length"] = df_chunk.apply(
            lambda x: compute_user_description_text_length(x) if (pd.notnull(x["user_description"])) else 0, axis=1)

        # Extract Tweet "entities" fields.
        df_chunk["tweet_entities_expanded_urls"] = df_chunk.apply(compute_expanded_urls, axis=1)
        df_chunk['tweet_entities_hashtags'] = df_chunk.apply(compute_hashtags, axis=1)
        df_chunk["tweet_entities_user_mentions_id"] = df_chunk.apply(compute_user_mentions_id, axis=1)
        df_chunk["tweet_entities_user_mentions_name"] = df_chunk.apply(compute_user_mentions_name, axis=1)
        df_chunk["tweet_entities_user_mentions_screen_name"] = df_chunk.apply(compute_user_mentions_screen_name, axis=1)
        df_chunk["tweet_entities_symbols"] = df_chunk.apply(compute_symbols, axis=1)

        # Create/update/infer fields. (original extracted/derived fields)
        df_chunk['retweeted_derived'] = df_chunk.apply(compute_retweet, axis=1)
        df_chunk['text_derived'] = df_chunk.apply(compute_full_text, axis=1)
        df_chunk['company_derived'] = df_chunk.apply(compute_company, axis=1)
        df_chunk['tweet_url_link_derived'] = df_chunk.apply(compute_url_link, axis=1)

        # Count the # of companies each Tweet is associated with.
        df_chunk['multiple_companies_derived_count'] = \
            df_chunk.apply(compute_multiple_companies, axis=1)

        # Determine whether Tweet is associated with "company_name" or "multiple" companies.
        df_chunk["company_derived_designation"] = df_chunk.apply(compute_company_designation, axis=1)

        # Compute Tweet text length.
        df_chunk["tweet_text_length_derived"] = df_chunk.apply(compute_text_length, axis=1)

        # Extract Tweet object "retweeted_status" object fields.
        df_chunk[retweeted_status_object_fields] = df_chunk.apply(compute_flatten_retweeted_status_attribute, axis=1)

        # Flatten nested fields in "retweeted_status_user".
        # FIXME - non-functional.
        # df_chunk[retweeted_status_user_object_fields] = df_chunk.apply(
        #     compute_flatten_retweeted_status_user_attributes, axis=1)

        # Polyglot only works on Linux (PITA to get working on Windows - sometimes impossible)
        # df_chunk['polyglot_lang_detect_non_english_only'] = df_chunk.apply(update_language_non_english_tweets, axis=1)
        # df_chunk['polyglot_lang_detect_all_tweets'] = df_chunk.apply(update_language_all_tweets(), axis=1)

        # Determine the Tweet text's language using spaCy natural language processing library. (note: slow)
        df_chunk["spaCy_language_detect_all_tweets"] = df_chunk.apply(
            lambda x: what_language(x) if (pd.notnull(x["tweet_full_text"])) else "none", axis=1)

        # Remove irrelevant tweets (non-English or unknown-company).
        # TODO - change to spacy instead of polyglot.
        if drop_irrelevant_tweets:
            df_chunk = df_chunk[
                ((df_chunk['company'] != '') &
                 (df_chunk['lang'].str.startswith('en') |
                  df_chunk['language_polyglot'].str.startswith('en')))
            ]

        # # Write each chunk to the combined dataset file.
        # df_chunk[required_fields].to_csv(f"{dataset_filepath}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC,
        #                                  mode='a', header=include_header)
        #
        # # Write select attributes within each chunk to a separate dataset file to reduce file size.
        # df_chunk[extra_fields].to_csv(f"{dataset_filepath}-extra.csv", index=False,
        #                               quoting=csv.QUOTE_NONNUMERIC, mode='a', header=include_header)

        # Write each chunk to the combined dataset file.
        df_chunk[required_fields].to_csv(dataset_filepath + ".csv", index=False, quoting=csv.QUOTE_NONNUMERIC,
                                         mode='a', header=include_header)

        # Write select attributes within each chunk to a separate dataset file to reduce file size.
        df_chunk[extra_fields].to_csv(dataset_filepath + "-extra.csv", index=False,
                                      quoting=csv.QUOTE_NONNUMERIC, mode='a', header=include_header)

        # Print a progress message.
        count += df_chunk.shape[0]
        # Only include the header once, at the top of the file.
        include_header = False
        # log.info(f'\t\tprocessed {count} records...')
        log.info("\t\tprocessed " + str(count) + " records...")

        # Debug purposes.
        # break

    # # Drop duplicate rows/examples/Tweets.
    # df_full = pd.read_csv(f"{dataset_filepath}.csv", sep=',', encoding="utf-8")
    # df_full.drop_duplicates(inplace=True)
    # # df_full.dropna(how="all")
    # df_full.to_csv(f"{dataset_filepath}.csv",
    #                index=False, header=True, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    # df_full.to_json(f"{dataset_filepath}.json", orient='records', lines=True)
    #
    # df_extra = pd.read_csv(f"{dataset_filepath}-extra.csv", sep=',', encoding="utf-8")
    # df_extra.drop_duplicates(inplace=True)
    # df_extra.to_csv(f"{dataset_filepath}-extra.csv",
    #                 index=False, header=True, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    # df_extra.to_json(f"{dataset_filepath}-extra.json", orient='records', lines=True)

    # Drop duplicate rows/examples/Tweets.
    df_full = pd.read_csv(dataset_filepath + ".csv", sep=',', encoding="utf-8")
    df_full.drop_duplicates(inplace=True)
    # df_full.dropna(how="all")
    df_full.to_csv(dataset_filepath + ".csv",
                   index=False, header=True, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    df_full.to_json(dataset_filepath + ".json", orient='records', lines=True)

    df_extra = pd.read_csv(dataset_filepath + "-extra.csv", sep=',', encoding="utf-8")
    df_extra.drop_duplicates(inplace=True)
    df_extra.to_csv(dataset_filepath + "-extra.csv",
                    index=False, header=True, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    df_extra.to_json(dataset_filepath + "-extra.json", orient='records', lines=True)

    # log.info(f'\tsaved the dataset to {dataset_filepath}'
    #          f'\n\t\tunknown company count: {unknown_company_count_global}'
    #          f'\n\t\tnon-English count: {non_english_count_global}'
    #          )
    log.info("\tsaved the dataset to " + dataset_filepath +
             "\n\t\tnon-English count: " + str(unknown_company_count_global) +
             "\n\t\tnon-English count: " + str(non_english_count_global)
             )


#########################################################################################################

def rename_column(row):
    """
    Function renames the column to something else.
    :param row: example in the dataset we are operating on.
    :return:  the modified example.
    """
    series = pd.read_json(json.dumps(row["quoted_status_id"]), typ='series')
    series = pd.Series(series)
    series_string = series.to_string()
    if len(series_string) > 0:
        return row["quoted_status_id"]
    row["quoted_status_id"] = np.NaN
    return row["quoted_status_id"]

    # if not pd.isnull(row["quoted_status_id"]):
    #     return row["quoted_status_id"]
    # row["quoted_status_id"] = np.NaN
    # return row["quoted_status_id"]


#########################################################################################################

def compute_user_series(row):
    """This function grabs the attributes specified in the List from the
    nested user JSON structure.
    """
    user_original_field_names = [
        'id', 'name', 'screen_name', 'location', 'description', 'followers_count', 'friends_count',
        'listed_count', 'favourites_count', 'statuses_count', 'created_at', 'time_zone', 'lang']

    user_series = pd.read_json(json.dumps(row['user']), typ='series')
    user_series['description'] = clean_text(user_series['description'])
    return user_series[user_original_field_names]


#########################################################################################################

def compute_user_description_text_length(row):
    """
    Function that determines the length of the user description text.
    :param row: example in the dataset we are operating on.
    :return:  the modified example.
    """
    row["user_description_text_length"] = len(row['user_description'])
    return row["user_description_text_length"]


#########################################################################################################

def compute_flatten_retweeted_status_attribute(row):
    """
    Function to extract 'retweeted_status' nested attributes from each example in the dataframe.
    :param row: current example (row) passed in.
    :return: nested attributes as individual columns added to the current example (row).
    """
    retweeted_status_original_field_names = [
        'created_at', 'id', 'full_text', 'in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name',
        'retweet_count', 'favorite_count', 'lang', 'entities', 'user', 'coordinates', 'place']

    if not pd.isnull(row["retweeted_status"]):
        series = pd.read_json(json.dumps(row["retweeted_status"]), typ='series')
        return series[retweeted_status_original_field_names]
    row[retweeted_status_original_field_names] = np.NaN
    return row[retweeted_status_original_field_names]


#########################################################################################################

def compute_flatten_retweeted_status_user_attributes(row):
    """
    Function to extract 'retweeted_status' nested "user" attributes from each example in the dataframe.
    :param row: current example (row) passed in.
    :return: nested attributes as individual columns added to the current example (row).
    FIXME - non-functional.
    """
    retweeted_status_original_user_field_names = [
        'id', 'name', 'screen_name', 'location', 'description', 'followers_count', 'friends_count',
        'listed_count', 'favourites_count', 'statuses_count', 'created_at', 'time_zone', 'lang']

    if not pd.isnull(row["retweeted_status_user"]):
        series = pd.read_json(json.dumps(row["retweeted_status_user"]), typ='series')
        return series[retweeted_status_original_user_field_names]
    # So, context-sensitive menus will give us available function calls.
    # row = pd.Series(row)
    # row.append(pd.Series(retweeted_status_user_object_fields), ignore_index=True)
    # print(f"{row}")
    row[retweeted_status_original_user_field_names] = np.NaN
    return row[retweeted_status_original_user_field_names]


#########################################################################################################

def compute_url_link(row):
    """This function constructs a URL referencing the full context of the
    given tweet.
    """
    # return f'https://twitter.com/-/status/{row["id"]}'
    return "https://twitter.com/-/status/" + str(row["id"])


#########################################################################################################

def compute_retweet(row):
    """This function determines if a tweet is a retweet."""
    return row['full_text'].startswith('RT')


#########################################################################################################

def compute_full_text(row):
    """This function creates the full text, either from the tweet itself or,
    if the tweet is a retweet (RT) that has been truncated (... at the end),
    by pasting the retweet header onto the original tweet text.
    """
    full_text = row['full_text']
    # If needed, reconstruct the full tweet text from the original text, leaving the retweet header intact.
    if full_text.startswith('RT @') and full_text.endswith('\u2026') and not pd.isnull(row['retweeted_status']):
        text_header = PTN_rt.search(row['full_text']).group()
        retweet_series = pd.read_json(json.dumps(row['retweeted_status']), typ='series')
        # full_text = f'{text_header}{retweet_series["full_text"]}'
        full_text = text_header + retweet_series["full_text"]
    return clean_text(full_text)


#########################################################################################################

def compute_text_length(row):
    """
    Function determines the length of the Tweet's text.
    :param row: current example in the dataframe we are operating on.
    :return: current example we are operating on.
    """
    derived_series = pd.read_json(json.dumps(row['text_derived']), typ='series')
    derived_series = pd.Series(derived_series)
    row["tweet_text_length_derived"] = derived_series.str.len()
    return row["tweet_text_length_derived"]


#########################################################################################################

def compute_expanded_urls(row):
    """
    Function extracts the expanded URL's from within the "entities" object.
    :param row: current example in the dataframe we are operating on.
    :return: current example we are operating on.
    """
    entity_series = pd.read_json(json.dumps(row['entities']), typ='series')
    expanded_urls = list(map(lambda entry: entry['expanded_url'], entity_series['urls']))
    return ','.join(expanded_urls)


#########################################################################################################

def compute_hashtags(row):
    """This function grabs the list of hashtags from the nested entities
    JSON structure.
    """
    entity_series = pd.read_json(json.dumps(row['entities']), typ='series')
    hashtags = list(map(lambda entry: entry['text'], entity_series['hashtags']))
    return ','.join(hashtags)


#########################################################################################################

def compute_user_mentions_name(row):
    """
    This function extracts the nested attributes within the "entities" object of the main Tweet object.
    :param row: a Tweet (line) from the raw JSON file.
    :return: the extracted attributes as individual columns.
    """
    entity_series = pd.read_json(json.dumps(row['entities']), typ='series')
    user_mentions_name = list(map(lambda entry: entry['name'], entity_series['user_mentions']))
    return ','.join(user_mentions_name)


def compute_user_mentions_screen_name(row):
    """
    This function extracts the nested attributes within the "entities" object of the main Tweet object.
    :param row: a Tweet (line) from the raw JSON file.
    :return: the extracted attributes as individual columns.
    """
    entity_series = pd.read_json(json.dumps(row['entities']), typ='series')
    user_mentions_screen_name = list(map(lambda entry: entry['screen_name'], entity_series['user_mentions']))
    return ','.join(user_mentions_screen_name)


def compute_user_mentions_id(row):
    """
    This function extracts the nested attributes within the "entities" object of the main Tweet object.
    :param row: a Tweet (line) from the raw JSON file.
    :return: the extracted attributes as individual columns.
    """
    entity_series = pd.read_json(json.dumps(row['entities']), typ='series')
    user_mentions_id = list(map(lambda entry: entry['id'], entity_series['user_mentions']))
    return user_mentions_id


#########################################################################################################

def compute_symbols(row):
    """This function grabs the list of hashtags from the nested entities
    JSON structure.
    """
    entity_series = pd.read_json(json.dumps(row['entities']), typ='series')
    user_mentions_symbols = list(map(lambda entry: entry['text'], entity_series['symbols']))
    return ','.join(user_mentions_symbols)


#########################################################################################################

def compute_company(row):
    """This function identifies the target company from the tweet text, issuing
    a warning for unrecognized texts. It assumes that the full, untruncated
    tweet text has already been constructed (see compute_full_text()).
    """
    global unknown_company_count_global
    associated_company = []

    # Identify the target company using known patterns in the tweet text.
    tweet = row['text_derived'].lower()
    author = row['user_screen_name'].lower()
    for company_pattern in PTN_companies:
        if re.compile(author).fullmatch(company_pattern[2]):
            associated_company.append(company_pattern[0])
            break
        if company_pattern[1].search(tweet):
            associated_company.append(company_pattern[0])

    if len(associated_company) > 0:
        return '|'.join(associated_company)

    # No company pattern applies, so it's unclear how this tweet was selected.
    unknown_company_count_global += 1
    # log.warning(f"\t\t\tunrecognized company (will be dropped): "
    #             f"\n\t\t\t\tid: {row['tweet_id']}"
    #             f"\n\t\t\t\ttweet: {row['text_derived']}"
    #             f"\n\t\t\t\thashtags: {row['tweet_entities_hashtags']}")
    # log.warning("\t\t\tunrecognized company (will be dropped): " +
    #             "\n\t\t\t\tid: " + str(row['tweet_id']) +
    #             "\n\t\t\t\ttweet: " + row['text_derived'] +
    #             "\n\t\t\t\thashtags: " + row['tweet_entities_hashtags'])
    return ''


#########################################################################################################

def compute_multiple_companies(row):
    """
    Function determines the number of companies a Tweet is associated with.
    Note: we convert derived_series to a series to avoid Pandas warning.
    :param row: example in the dataset we are operating on.
    :return:  the modified example.
    TODO - check we have fixed our logic error!
    """
    derived_series = pd.read_json(json.dumps(row['company_derived']), typ='series')
    derived_series = pd.Series(derived_series)
    derived_string = derived_series.to_string()
    if derived_string.count('|') > 0:
        row["multiple_companies_derived_count"] = derived_string.count('|') + 1
    elif derived_string != "Series([], )":
        row["multiple_companies_derived_count"] = 1
    else:
        row["multiple_companies_derived_count"] = 0
    return row["multiple_companies_derived_count"]


#########################################################################################################

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


#########################################################################################################

def update_language_non_english_tweets(row):
    """This function computes an alternate language code for the given
    tweet using TextBlob, a more reliable language coder.
    """
    global non_english_count_global
    if row['lang'].startswith('en'):
        # Leave English codes (i.e., en, en-gb) unchanged.
        return row['lang']
    else:
        # Compute alternate code for non-English tweets, many of which are
        # in English as well.
        lang2 = Text(row['full_text']).language.code
        if not lang2.startswith('en'):
            non_english_count_global += 1
            # log.warning(f"\t\t\tnon-English tweet (will be dropped): "
            #             f"\n\t\t\t\tid: {row['tweet_id']}"
            #             f"\n\t\t\t\ttweet: {row['text_derived']}"
            #             f"\n\t\t\t\tLanguage tags: {row['tweet_lang']} - {lang2}"
            #             )
            log.warning("\t\t\tnon-English tweet (will be dropped): " +
                        "\n\t\t\t\tid: " + str(row['tweet_id']) +
                        "\n\t\t\t\ttweet: " + str(row['text_derived']) +
                        "\n\t\t\t\tLanguage tags: " + row['tweet_lang'] + "-" + lang2
                        )
        return lang2


def update_language_all_tweets(row):
    """This function computes an alternate language code for the given
    tweet using TextBlob, a more reliable language coder.
    """
    global non_english_count_global
    # Compute alternate code for all Tweets.
    lang2 = Text(row['full_text']).language.code
    if not lang2.startswith('en'):
        non_english_count_global += 1
        # log.warning(f"\t\t\tnon-English tweet (will be dropped): "
        #             f"\n\t\t\t\tid: {row['tweet_id']}"
        #             f"\n\t\t\t\ttweet: {row['text_derived']}"
        #             f"\n\t\t\t\tLanguage tags: {row['tweet_lang']} - {lang2}"
        #             )
        log.warning("\t\t\tnon-English tweet (will be dropped): " +
                    "\n\t\t\t\tid: " + str(row['tweet_id']) +
                    "\n\t\t\t\ttweet: " + str(row['text_derived']) +
                    "\n\t\t\t\tLanguage tags: " + row['tweet_lang'] + "-" + lang2
                    )
    return lang2


#########################################################################################################

def what_language(row):
    """
     Function utilizes spaCy N.L.P. library, "langdetect" library, and "spacy-langdetect" library
     to determine the language of the Tweet.
    :param row: example in the dataset we are operating on.
    :return:  the modified example with additional column specifying its language.
    """
    nlp = spacy.load("en")
    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    document = nlp(row["tweet_full_text"])
    # document level language detection. Think of it like average language of document!
    text_language = document._.language
    row["spaCy_language_detect"] = str(text_language["language"])
    return row["spaCy_language_detect"]


#########################################################################################################

def clean_text(text):
    """Do simple text cleanup for the data processing files."""
    return text.replace('\n', ' ').replace('\r', ' ')


#########################################################################################################

def remove_filepath_if_exists(filepath):
    """Delete the given file if it exists."""
    if os.path.isfile(filepath):
        # log.info(f'\tdeleting existing file: {filepath}')
        log.info("\tdeleting existing file: " + filepath)
        os.remove(filepath)


#########################################################################################################

def create_separate_company_datasets(dataset_filepath, dataset_path, filename_base):
    """Read the given full/combined dataset file and create/save
    company-specific groups.
    """
    # log.info(f'\tsplitting dataset into company-specific datasets...')
    log.info("\tsplitting dataset into company-specific datasets...")
    df = pd.read_csv(dataset_filepath, encoding='utf-8', engine='python')
    for company_name, group in df.groupby(['company_derived']):
        group.to_csv(
            # dataset_path / f'{filename_base}-{company_name}.csv',
            dataset_path / filename_base + "-" + company_name + ".csv",
            index=False)


#########################################################################################################
#########################################################################################################

def main(json_data_filepath='dataset.json',
         dataset_path='.',
         filename_base='dataset',
         drop_irrelevant_tweets=True,
         add_company_datasets=False,
         logging_level=log.INFO,
         ):
    """This tool loads the raw JSON-formatted tweets from the given
    filepath, does some general updates to the dataset items and saves
    the results in filename (.csv). The columns are modified as
    follows:

    - The tweet text is modified to remove newlines (\\n, \\r).
    - A company column is added to identify the company of the tweet.
    - A language_textblob column is added to give a second (more accurate)
      language tag.

    Keyword Arguments:
        json_data_filepath -- the system path from which to load the raw JSON files
            (default='.')
        dataset_path -- the system path to which the dataset files are to be written
            (default='.')
        filename_base -- the base name of a/the pre-saved dataset files
            (default='dataset')
        encoding -- the file encoding to use
            (default: 'utf-8')
        add_company_datasets -- whether to add company-specific datasets
            (default: True)
        logging_level -- the level of logging to use
            (default: logging.INFO)

    TODO - we are not currently using the "Fire" library for CLI operation.
    """
    log.basicConfig(level=logging_level, format='%(message)s')
    # log.info(f'building the dataset')
    log.info("building the dataset")

    if not os.path.isfile(json_data_filepath):
        # log.fatal(f'\tfilepath doesn\'t exist: {json_data_filepath}')
        log.fatal("\tfilepath doesn\'t exist: " + json_data_filepath)
        exit(-1)

    # full_dataset_filepath = Path(dataset_path) / f'{filename_base}.csv'
    full_dataset_filepath = Path(dataset_path) / filename_base + ".csv"
    remove_filepath_if_exists(full_dataset_filepath)

    create_dataset(Path(json_data_filepath), full_dataset_filepath, drop_irrelevant_tweets)

    if add_company_datasets:
        create_separate_company_datasets(full_dataset_filepath,
                                         Path(dataset_path),
                                         filename_base)


#########################################################################################################

def check_for_preexisting_output_file(output_file_path):
    """
    This function checks for any pre-existing file that has the same name as the specified output file name for the
    dataset at the specified save location.  If so, it will ABORT the operation to ensure that we are not appending to
    a pre-existing CSV file.

    Resources:
    https://www.guru99.com/python-check-if-file-exists.html

    :return: None.
    """

    # if path.exists(f"{output_file_path}"):
    if path.exists(output_file_path):
        print("Output file at specified save location file path already exists!")
        print("Aborting operation!")
        sys.exit()


#########################################################################################################
#########################################################################################################

if __name__ == '__main__':
    # Fire(main)
    # Example invocation:
    # python dataset_processor.py
    # --json_data_filepath=/media/hdd_2/slo/stance/slo-tweets-20160101-20180304/dataset.json
    # --dataset_path=/media/hdd_2/slo/stance/datasets

    start_time = time.time()

    # Absolute file path.
    create_dataset("/home/jj47/Summer-Research-2019-master/json/dataset_slo_20100101-20180510.json",
                   "/home/jj47/Summer-Research-2019-master/twitter-dataset-6-27-19",
                   False)
    end_time = time.time()

    time_elapsed_seconds = (end_time - start_time)
    time_elapsed_minutes = (end_time - start_time) / 60.0
    time_elapsed_hours = (end_time - start_time) / 60.0 / 60.0
    # print(f"Time taken to process dataset: {time_elapsed_seconds} seconds, "
    #       f"{time_elapsed_hours} hours, {time_elapsed_minutes} minutes")
    print("Time taken to process dataset: " + str(time_elapsed_seconds) + " seconds, " +
          str(time_elapsed_hours) + "hours, " + str(time_elapsed_minutes) + " minutes")

#########################################################################################################
#########################################################################################################
