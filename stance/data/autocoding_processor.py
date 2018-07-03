import datetime
import fire
import re

import pandas as pd

import logging
logger = logging.getLogger(__name__)

from data.settings import PTN_against_hashtags, PTN_for_hashtags, PTN_neutral_screennames, PTN_for_screennames


def get_size(df):
    """Get the number of rows in the given dataframe."""
    return df.shape[0]


def main(dataset_filepath='dataset.csv',
         testset_filepath=None,
         encoding='utf-8',
         coding_filepath='.',
         logging_level: int = logging.INFO,
         for_sample_size: int = None,
         against_sample_size: int = None,
         neutral_sample_size: int = None,
         companytweets=False
         ):
    """This function creates a auto-coded dataset using distance supervision,
    for Adani only, using simple hashtag rules. The three stance codings are
    balanced based on the number of "for" tweets.

    # Args

    dataset_filepath:
        csv file path (default: 'dataset.csv')
    encoding:
        file character encoding (default: 'utf-8')
    coding_path:
        path in which to dump the output file (default: '.')
    logging_level
        the level of logging to use (default: logging.INFO)
    against_multiplier
        the ratio of against to for tweets to be sampled
    neutral_multiplier
        the ratio of neutral to for tweets to be sampled
    """
    logging.basicConfig(level=logging_level, format='%(message)s')
    logger.info(f'auto-coding tweets')

    if testset_filepath:
        logger.info(f'\tloading testset file: {testset_filepath}')
        df_testset = pd.read_csv(testset_filepath, encoding=encoding, engine='python')
        test_ids = re.compile('|'.join(pd.Series(df_testset['id']).apply(str)))

    logger.info(f'\tloading dataset file: {dataset_filepath}')
    df_all = pd.read_csv(dataset_filepath, encoding=encoding, engine='python')
    logger.info(f'\t\t{get_size(df_all)} items loaded')
    if testset_filepath:
        df_adani = df_all.loc[(df_all['company'] == 'adani') & (~df_all['id'].astype(str).str.match(test_ids))]
    else:
        df_adani = df_all.loc[(df_all['company'] == 'adani')]
    logger.info(f'\t\t{get_size(df_adani)} adani items loaded')

    # Replace all semicolons to fix column displacement issues.
    df_adani['text'] = df_adani['text'].str.replace(";", "")
    df_adani['user_description'] = df_adani['user_description'].str.replace(";", "")

    # Annotate tweets with suspected stance values using rule patterns.
    if companytweets:
        df_adani['auto_for'] = (df_adani['text'].str.lower().str.contains(PTN_for_hashtags) |
                                (~df_adani['text'].str.lower().str.contains(PTN_for_hashtags) &
                                df_adani['user_screen_name'].str.lower().str.match(PTN_for_screennames)))
    else:
        df_adani['auto_for'] = df_adani['text'].str.lower().str.contains(PTN_for_hashtags)
    df_adani['auto_against'] = df_adani['text'].str.lower().str.contains(PTN_against_hashtags)
    df_adani['auto_neutral'] = df_adani['user_screen_name'].str.lower().str.match(PTN_neutral_screennames)

    # Collect tweets that are to be coded for each stance value.
    if for_sample_size:
        df_for = df_adani.loc[df_adani['auto_for'] & ~df_adani['auto_against'] & ~df_adani['auto_neutral']
                        & ~df_adani['retweeted']].sample(for_sample_size)
    else:
        df_for = df_adani.loc[df_adani['auto_for'] & ~df_adani['auto_against'] & ~df_adani['auto_neutral']
                        & ~df_adani['retweeted']].sample(frac=1.0)
    df_for['stance'] = 'for'
    df_for['confidence'] = 'auto'
    if not against_sample_size:
        against_sample_size = df_for.shape[0]
    df_against = df_adani.loc[~df_adani['auto_for'] & df_adani['auto_against'] & ~df_adani['auto_neutral']
                        & ~df_adani['retweeted']].sample(against_sample_size)
    df_against['stance'] = 'against'
    df_against['confidence'] = 'auto'
    if not neutral_sample_size:
        neutral_sample_size= df_for.shape[0]
    df_neutral = df_adani.loc[~df_adani['auto_for'] & ~df_adani['auto_against'] & df_adani['auto_neutral']
                        & ~df_adani['retweeted']].sample(neutral_sample_size)
    df_neutral['stance'] = 'neutral'
    df_neutral['confidence'] = 'auto'
    # ambiguous_df = df.loc[(df['auto_for'] & df['auto_against']) |
    #                       (~df['auto_for'] & ~df['auto_against'] & ~df['auto_neutral'])]

    # Remove the auto_* fields because they are useful only for computing the stance column value.
    df_adani.drop(columns=['auto_for', 'auto_against', 'auto_neutral'])

    # Save the auto-coded items in one file.
    autocoded_dataset_filepath = f'{coding_filepath}'
    logger.info(f'\tstoring auto-coded dataset file: {autocoded_dataset_filepath}')
    pd.concat([df_for, df_against, df_neutral]).to_csv(autocoded_dataset_filepath)
    print(f'\t\tfor: {get_size(df_for)}\n\t\tagainst: {get_size(df_against)} (sampled)\n\t\tneutral: {get_size(df_neutral)} (sampled)')


if __name__ == '__main__':
    fire.Fire(main)
    # Example invocation:
    # python autocoding_processor.py --dataset_filepath=/media/hdd_2/slo/stance/datasets/dataset.csv --coding_filepath=/media/hdd_2/slo/stance/coding --against_muliplier=1 --for_multiplier=1
