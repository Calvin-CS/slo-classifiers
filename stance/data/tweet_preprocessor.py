"""Preprocess tweets for machine learning input."""
import csv
import html

from fire import Fire
import pandas as pd

import logging
logger = logging.getLogger(__name__)

import settings
from vendor import CMUTweetTagger as Tagger


def preprocess_text(text: str) -> str:
    """Preprocess Tweets.

    - RT sign (`RT @mention: `)
    - shrink elongations:
        - letters -- waaaaaay -> waaay
        - signs   -- !!!!!!!! -> !!!
    - downcase
    - replace newline (and tab) with a space
    - URLs:
        - remove truncated URL fragments
        - add space between [.,?!] and 'http'
    - fix HTML escaped tokens
    - abstract numericals:
        - 1964  -> year
        - 16:32 -> time
        - $12   -> money


    To abstract mentions and urls depends on the purpose (for ML model input or for creating word embeddings).
    So this function does not do such abstracting.
    """
    try:
        # TODO: Why should these fields ever be Nan? Fix this in dataset_processor.py?
        if type(text) == float:
            text = ""
        text = html.unescape(text)
        text = settings.PTN_rt.sub('', text)
        text = settings.PTN_whitespace.sub(' ', text)
        text = settings.PTN_concatenated_url.sub(r'\1 http', text)

        # preserve Twitter specific tokens
        # username can contain year notations and elongations
        mentions = settings.PTN_mention.findall(text)
        SLO_MENTION_PLACEHOLDER = r'slo_mention'
        text = settings.PTN_mention.sub(SLO_MENTION_PLACEHOLDER, text)
        # URLs might be case sensitive
        urls = settings.PTN_url.findall(text)
        SLO_URL_PLACEHOLDER = r'slo_url'
        text = settings.PTN_url.sub(SLO_URL_PLACEHOLDER, text)

        text = settings.PTN_elongation.sub(r'\1\1\1', text)
        text = text.lower()

        text = settings.PTN_year.sub('slo_year', text)
        text = settings.PTN_time.sub('slo_time', text)
        text = settings.PTN_cash.sub('slo_cash', text)

        # put back Twitter specific tokens
        for url in urls:
            text = text.replace(SLO_URL_PLACEHOLDER, url, 1)
        for mention in mentions:
            text = text.replace(SLO_MENTION_PLACEHOLDER, mention, 1)

    except:
        logger.error(f'preprecessing error on: {text}; {type(text)}')
        raise

    return text


def postprocess_text(text: str) -> str:
    """Postprocess an input text.

    This function modifies an input text as the precise input to word embedding tools

    - abstract mentions and URLs

    This does not touch hashtags and cashtags because treating them as different words will work for our task.
    """
    text = settings.PTN_mention.sub(r'slo_mention', text)
    text = settings.PTN_url.sub(r'slo_url', text)
    return text


def read_dataset(fp: str, extension: str, encoding: str) -> pd.DataFrame:
    """This function reads the specified dataset, in whichever format."""
    logger.info(f'\tloading dataset file: {fp}')
    if extension == 'csv':
        df = pd.read_csv(fp, encoding=encoding, engine='python')
    elif extension == 'json':
        df = pd.read_json(fp)
    else:
        raise Exception(f'file {fp} not valid - only CSV and JSON accepted...')
    logger.info(f'\t\t{df.shape[0]} items loaded')
    return df


def save_datasets(df: pd.DataFrame, filepath: str, separate_companies: bool) -> None:
    """This function saves the tokenized datasets, one for each company and
    one for all the companies combined.
    """
    csv_filename = f'{filepath}_tok.csv'
    df.to_csv(csv_filename, index=False)
    logger.info(f'\t\t{csv_filename} - {df.shape[0]} items')
    if separate_companies:
        for company_name, group in df.groupby('company'):
            csv_filename = f'{filepath}-{company_name}_tok.csv'
            group.to_csv(csv_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
            logger.info(f'\t\t{csv_filename} - {group.shape[0]} items')


def fix_for_tagger(texts):
    """The CMU tokenizer/tagger doesn't handle empty ('') texts properly.
    Hack this by replacing them with 'PLACEHOLDER'. Tweets are never (?)
    empty, but user profile descriptions are frequently empty.
    """
    return [text if text != '' else 'slo_empty_text' for text in texts]


def main(fp: str='dataset.csv',
         tweet_column_name: str='text',
         profile_column_name='user_description',
         encoding: str='utf-8',
         to_csv: bool=True,
         separate_companies: bool=False,
         postproc: bool=True,
         logging_level: int=logging.INFO
         ) -> None:
    """Preprocess and tokenise.

    This tool only runs on Linux/OSx, not Windows.

    1. Load a csv which contains tweets
    2. Preprocess tweets by `preprocess_text`
    3. Tokenise preprocessed tweets by CMUTweetTagger
    4. Print tokenised tweets on stdout or write another csv in which tokenised tweets are added to a column named 'tweet_t'

    # Args

    fp:
        csv or json file path
        (default: 'dataset.csv')
    tweet_column_name:
        the column name of tweets in the input csv
        (default: 'text')
    profile_column_name:
        the column name of author profile description in the input csv
        (default: 'user_description')
    encoding:
        file character encoding
        (default: 'utf-8')
    to_csv:
        if True, the latter process in Procedure 4 is executed
        (default: True)
    separate_companies:
        if True, separate files grouped by company name are produced
        (default: False)
    postproc:
        if True, further conduct postprocessing
        (default: True)
    logging_level
        the level of logging to use
        (default: logging.INFO)
    """
    logging.basicConfig(level=logging_level, format='%(message)s')
    logger.info(f'processing tweets')

    filename, extension = fp.split('.')
    df = read_dataset(fp, extension, encoding)

    logger.info(f'\tpre-processing tweets and profile descriptions...')
    tweets_p = df[tweet_column_name].apply(preprocess_text)
    profiles_p = df[profile_column_name].apply(preprocess_text)

    logger.info(f'\tparsing/tagging tweets...')
    tagged = Tagger.runtagger_parse(fix_for_tagger(tweets_p))
    tweets_t = pd.Series([
        ' '.join([w for w, pos, p in row]) for row in tagged
    ])
    tagged = Tagger.runtagger_parse(fix_for_tagger(profiles_p))
    profiles_t = pd.Series([
        ' '.join([w for w, pos, p in row]) for row in tagged
    ])

    if postproc:
        logger.info(f'\tpost-processing tweets...')
        tweets_t = tweets_t.apply(postprocess_text)
        profiles_t = profiles_t.apply(postprocess_text)

    if to_csv:
        df['tweet_t'] = tweets_t
        df['profile_t'] = profiles_t
        logger.info(f'\tsaving processed tweets and profiles:')
        save_datasets(df, filename, separate_companies)
    else:
        for tweet in tweets_t:
            print(tweet)
        for profile in profiles_t:
            print(profile)


if __name__ == '__main__':
    Fire(main)
    # Example invocation:
    # python tweet_preprocessor.py --fp=/media/hdd_2/slo/stance/datasets/dataset.csv --to_csv --postproc
