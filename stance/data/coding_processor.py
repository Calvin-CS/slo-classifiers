import logging
import datetime
import fire
import requests
import re
from pathlib import Path

import numpy as np
import pandas as pd
from textblob import TextBlob


# Regex for Twitter mentions (e.g., @kvlinden)
mention_re = re.compile('\s*@\w{1,15}\s*')


def load_prepared_dataset(path_filename_base, encoding):
    """This function loads an existing dataset."""
    full_path = f'{path_filename_base}.csv'
    logging.info(f'\tloading dataset file: {full_path}')
    # Use the python engine because it is more complete (but slower).
    return pd.read_csv(full_path, encoding=encoding, engine='python')


def create_coding_set(path_filename_base, df, size, company_names, encoding):
    """This function creates coding sets and stores them in separate files."""
    file_handler = logging.FileHandler(f'{path_filename_base}.txt', mode='w')
    logging.getLogger().addHandler(file_handler)
    logging.info(f'building the coding set\n\t{datetime.datetime.now()}')
    coding_set = pd.DataFrame()
    for name, group in df.groupby('company'):
        if company_names == None or name in company_names:
            logging.info(f'\t{name} : {size} entries')
            coding_set = coding_set.append(get_sample_tweets(group, size))
    coding_columns = ['stance', 'confidence', 'value']
    data_columns = ['id', 'tweet_url', 'company', 'screen_name', 'text', 'user_description']
    for c in coding_columns:
        coding_set[c] = '??'
    coding_set.to_csv(f'{path_filename_base}.csv',
                      columns= coding_columns + data_columns,
                      encoding=encoding
                      )
    logging.getLogger().removeHandler(file_handler)


def get_sample_tweets(group, size):
    """This function samples the given number (size) of tweets from the given
    group that satisfy the coding requirements.
    """
    group_coding_set = pd.DataFrame()
    current_ids = set()
    while get_size(group_coding_set) < size:
        # Sample one English tweet that isn't a bare retweet or a BOT tweet.
        sample_tweet = group[
            # Accept original tweets.
            (group['retweeted']) &
            # Accept empty or small hashtag lists.
            ((group['hashtags'].str.len().isnull()) |
             (group['hashtags'].str.len() < 3)) &
            # (SLO_Dataset.get_string_list_len(group['hashtags']) < 6) &
            # Accept tweets coded as English by either Twitter or TextBlob.
            ((group['lang'].str.startswith('en')) |
             (group['language_textblob'].str.startswith('en')))
            ].sample(1)
        sample_tweet['tweet_url'] = sample_tweet.id.apply(
            create_tweet_url
        )
        # Skip this tweet if it is deleted/invalid or repeated.
        if check_tweet_accessibility(sample_tweet['tweet_url'].values[0]) and \
                not sample_tweet['id'].values[0] in current_ids:
            # Add this usable tweet with coding text to the coding set.
            sample_tweet['tweet_to_code'] = sample_tweet['text'].apply(
                remove_prepended_mentions)
            sample_tweet['stance'] = ''
            sample_tweet['confidence'] = ''
            group_coding_set = group_coding_set.append(sample_tweet)
            current_ids.add(sample_tweet['id'].values[0])
            # Show progress in finding appropriate tweets.
            #     Useful if there are relatively few appropriate tweets
            # logging.info(f'\t\t{SLO_Dataset.get_size(group_coding_set)}')
    return group_coding_set


def get_size(df):
    """This function gets the number of rows in the given dataframe."""
    return df.shape[0]


def check_tweet_accessibility(tweet_url):
    """This function determines whether the given tweet is still accessible
    via the Twitter REST API.

    Arguments:
        tweet_url -- the URL for the tweet
    """
    response = requests.get(tweet_url)
    # Accessible tweets give HTTP 200 and include the screen name and tweet
    # ID in the URL. Inaccessible tweets can give 404 responses or redirect
    # to account/suspended.
    return response.status_code == 200 and \
           response.url.find('suspended') == -1


def remove_prepended_mentions(tweet):
    start = 0
    while True:
        match = mention_re.match(tweet, start)
        if match == None:
            break
        start = match.end()
    return tweet[start:]


def create_tweet_url(tweet_id):
    """This function constructs a URL referencing the full context of the
    given tweet.
    """
    return f'https://twitter.com/-/status/{tweet_id}'


def main(data_path='.',
         path='.',
         dataset_filename_base='dataset',
         coding_filename_base='coding',
         size=10,
         company_names=None,
         encoding='utf-8',
         logging_level=logging.INFO,
         ):
    """This method selects a random set of tweets to code for each company of
    the given size. The tweets and a log of the creation process are stored
    in files using the given filename (.csv and .txt respectively). It
    assumes that the dataset has already been loaded. The columns are modified
    as follows:

    - A tweet_to_code column is added, which is the original tweet stripped of
      leading mentions.
    - A tweet_url column is added to give a direct Twitter URL for the tweet.
    - Empty columns are added for stance and confidence.

    The produced CSV file will include long integers that Excel doesn't handle
    by default. To solve the problem, import the .csv file as shown here:

    http://techsupport.matomy.com/Reports/29830196/How-to-present-long-numbers-correctly-in-Excel-CSV-file.htm

    Keyword arguments:
        data_path -- the system path from which to read the dataset
            (default: './data')
        path -- the system path to which to write the coding dataset files
            (default: '.')
        dataset_filename_base -- a base filename for the dataset file
            (default: 'dataset')
        coding_filename_base -- a base filename for the new coding set
            (default: 'coding')
        size -- the number of coding set elements to sample for each company
            (default: 10)
        company_names -- a list of company names for which to collect samples
            (default: None -- collect for all companies)
        encoding -- the file encoding to use
            (default: 'utf-8')
        logging_level -- the level of logging to use
            (default: logging.INFO)
    """
    logging.basicConfig(level=logging_level, format='%(message)s')
    df = load_prepared_dataset(Path(data_path) / dataset_filename_base, encoding)
    create_coding_set(Path(path) / coding_filename_base, df, size, company_names, encoding)


if __name__ == '__main__':
    fire.Fire(main)

    # Example invocation:
    # python coding_processor.py --data_path=/media/hdd_2/slo/data/slo-tweets-20160101-20180304 --path='/media/hdd_2/slo/data/coding' --size=50 --company_names=['adani']
