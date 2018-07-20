import os
import csv
import json
import re
from fire import Fire
from pathlib import Path

import numpy as np
import pandas as pd
from polyglot.text import Text, Word

import logging
logger = logging.getLogger(__name__)

from settings import PTN_rt, PTN_companies

# Count irrelevant tweets.
unknown_company_count_global = 0
non_english_count_global = 0

def create_dataset(json_data_filepath, dataset_filepath, encoding, drop_irrelevant_tweets):
    """This function rebuilds a dataset from the given raw JSON file.
    It reads/writes the (large) file in chunks.
    """
    global unknown_company_count_global, non_english_count_global
    logger.info(f'\tloading raw tweets from {json_data_filepath}')

    # Load/save the file in chunks.
    count = 0
    include_header = True
    for df_chunk in pd.read_json(json_data_filepath,
                                 orient='records',
                                 lines=True,
                                 chunksize=100000,
                                 # encoding=encoding,
                                 ):

        # Create/update/infer fields.
        df_chunk['retweeted'] = df_chunk.apply(compute_retweet, axis=1)
        df_chunk['text'] = \
            df_chunk.apply(compute_full_text, axis=1)
        df_chunk['language_polyglot'] = \
            df_chunk.apply(update_language, axis=1)
        df_chunk[['user_screen_name', 'user_description']] = \
            df_chunk.apply(compute_user_series, axis=1)
        df_chunk['hashtags'] = \
            df_chunk.apply(compute_hashtags, axis=1)
        df_chunk['company'] = df_chunk.apply(compute_company, axis=1)

        # Remove irrelevant tweets (non-English or unknown-company).
        if drop_irrelevant_tweets:
            df_chunk = df_chunk[
                (df_chunk['company'] != '') &
                ((df_chunk['lang'].str.startswith('en')))  # |
                 #(df_chunk['language_polyglot'].str.startswith('en')))
            ]

        # Write each chuck to the combined dataset file.
        required_fields = ['id', 'lang', 'language_polyglot',
                           'retweeted', 'hashtags', 'company', 'text',
                           'user_screen_name', 'user_description']
        df_chunk[required_fields].to_csv(dataset_filepath,
                                         index=False,
                                         quoting=csv.QUOTE_NONNUMERIC,
                                         mode='a',
                                         header=include_header,
                                         )

        # Print a progress message.
        count += get_size(df_chunk)
        # Only include the header once, at the top of the file.
        include_header = False
        logger.info(f'\t\tprocessed {count} records...')

    df_full = pd.read_csv(dataset_filepath)
    df_full.drop_duplicates(inplace=True)
    df_full.to_csv(dataset_filepath, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f'\tsaved the dataset to {dataset_filepath}'
                 f'\n\t\tunknown company count: {unknown_company_count_global}'
                 f'\n\t\tnon-English count: {non_english_count_global}'
                 )


def compute_retweet(row):
    """This function determines if a tweet is a retweet."""
    return row['full_text'].startswith('RT')


def compute_full_text(row):
    """This function creates the full text, either from the tweet itself or,
    if the tweet is a retweet (RT) that has been truncated (... at the end),
    by pasting the retweet header onto the original tweet text.
    """
    full_text = row['full_text']

    # If needed, reconstruct the full tweet text from the original text,
    # leaving the retweet header intact.
    if full_text.startswith('RT @') \
            and full_text.endswith('\u2026') \
            and not pd.isnull(row['retweeted_status']):
        text_header = PTN_rt.search(row['full_text']).group()
        retweet_series = pd.read_json(json.dumps(row['retweeted_status']),
                                      typ='series')
        full_text = f'{text_header}{retweet_series["full_text"]}'

    return clean_text(full_text)


def update_language(row):
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
            logger.warning(f"\t\t\tnon-English tweet (will be dropped): "
                            f"\n\t\t\t\tid: {row['id']}"
                            f"\n\t\t\t\ttweet: {row['text']}"
                            f"\n\t\t\t\tLanguage tags: {row['lang']} - {lang2}"
                            )
        return lang2


def compute_user_series(row):
    """This function grabs the user name and profile description from the
    nested user JSON structure.
    """
    user_series =  pd.read_json(json.dumps(row['user']), typ='series')
    user_series['description'] = clean_text(user_series['description'])
    return user_series[['screen_name', 'description']]


def compute_hashtags(row):
    """This function grabs the list of hashtags from the nested entities
    JSON structure.
    """
    entity_series =  pd.read_json(json.dumps(row['entities']), typ='series')
    hashtags = list(map(lambda entry: entry['text'], entity_series['hashtags']))
    return ','.join(hashtags)


def compute_company(row):
    """This function identifies the target company from the tweet text, issuing
    a warning for unrecognized texts. It assumes that the full, untruncated
    tweet text has already been constructed (see compute_full_text()).
    """
    global unknown_company_count_global

    associated_company = []

    # Identify the target company using known patterns in the tweet text.
    tweet = row['text'].lower()
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
    logger.warning(f"\t\t\tunrecognized company (will be dropped): "
                    f"\n\t\t\t\tid: {row['id']}"
                    f"\n\t\t\t\ttweet: {row['text']}"
                    f"\n\t\t\t\thashtags: {row['hashtags']}")
    return ''


def get_size(df):
    """Get the number of rows in the given dataframe."""
    return df.shape[0]


def clean_text(text):
    """Do simple text cleanup for the data processing files."""
    return text.replace('\n', ' ').replace('\r', ' ')


def remove_filepath_if_exists(filepath):
    """Delete the given file if it exists."""
    if os.path.isfile(filepath):
        logger.info(f'\tdeleting existing file: {filepath}')
        os.remove(filepath)


def create_separate_company_datasets(dataset_filepath, dataset_path, filename_base):
    """Read the given full/combined dataset file and create/save
    company-specific groups.
    """
    logger.info(f'\tsplitting dataset into company-specific datasets...')
    df = pd.read_csv(dataset_filepath, encoding='utf-8', engine='python')
    for company_name, group in df.groupby(['company']):
        group.to_csv(
            dataset_path / f'{filename_base}-{company_name}.csv',
            index=False)


def main(json_data_filepath='dataset.json',
         dataset_path='.',
         filename_base='dataset',
         encoding='utf-8',
         drop_irrelevant_tweets=True,
         add_company_datasets=False,
         logging_level=logging.INFO,
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
    """
    logging.basicConfig(level=logging_level, format='%(message)s')
    logger.info(f'building the dataset')

    if not os.path.isfile(json_data_filepath):
        logger.fatal(f'\tfilepath doesn\'t exist: {json_data_filepath}')
        exit(-1)

    full_dataset_filepath = Path(dataset_path) / f'{filename_base}.csv'
    remove_filepath_if_exists(full_dataset_filepath)

    create_dataset(Path(json_data_filepath), full_dataset_filepath, encoding, drop_irrelevant_tweets)

    if add_company_datasets:
        create_separate_company_datasets(full_dataset_filepath,
                                         Path(dataset_path),
                                         filename_base)


if __name__ == '__main__':
    Fire(main)
    # Example invocation:
    # python dataset_processor.py --json_data_filepath=/media/hdd_2/slo/stance/slo-tweets-20160101-20180304/dataset.json --dataset_path=/media/hdd_2/slo/stance/datasets
