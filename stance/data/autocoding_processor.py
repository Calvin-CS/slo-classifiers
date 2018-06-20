import datetime
import fire

import pandas as pd

import logging
logger = logging.getLogger(__name__)

from data.settings import PTN_against_hashtags, PTN_for_hashtags, PTN_neutral_screennames


def get_size(df):
    """Get the number of rows in the given dataframe."""
    return df.shape[0]


def main(dataset_filepath='dataset.csv',
         encoding='utf-8',
         coding_path='.',
         logging_level: int = logging.INFO,
         against_multiplier: int = 1,
         neutral_multiplier: int = 1,
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

    logger.info(f'\tloading dataset file: {dataset_filepath}')
    df_all = pd.read_csv(dataset_filepath, encoding=encoding, engine='python')
    logger.info(f'\t\t{get_size(df_all)} items loaded')
    df_adani = df_all.loc[df_all['company'] == 'adani']
    logger.info(f'\t\t{get_size(df_adani)} adani items loaded')

    # Replace all semicolons to fix column displacement issues.
    df_adani['text'] = df_adani['text'].str.replace(";", "")
    df_adani['user_description'] = df_adani['user_description'].str.replace(";", "")

    # Annotate tweets with suspected stance values using rule patterns.
    df_adani['auto_for'] = df_adani['text'].str.lower().str.contains(PTN_for_hashtags)
    df_adani['auto_against'] = df_adani['text'].str.lower().str.contains(PTN_against_hashtags)
    df_adani['auto_neutral'] = df_adani['user_screen_name'].str.lower().str.match(PTN_neutral_screennames)

    # Collect tweets that are to be coded for each stance value.
    df_for = df_adani.loc[df_adani['auto_for'] & ~df_adani['auto_against'] & ~df_adani['auto_neutral']
                    & ~df_adani['retweeted']][:]
    df_for['stance'] = 'for'
    df_for['confidence'] = 'auto'
    df_against = df_adani.loc[~df_adani['auto_for'] & df_adani['auto_against'] & ~df_adani['auto_neutral']
                        & ~df_adani['retweeted']].sample(get_size(df_for)*against_multiplier)
    df_against['stance'] = 'against'
    df_against['confidence'] = 'auto'
    df_neutral = df_adani.loc[~df_adani['auto_for'] & ~df_adani['auto_against'] & df_adani['auto_neutral']
                        & ~df_adani['retweeted']].sample(get_size(df_for)*neutral_multiplier)
    df_neutral['stance'] = 'neutral'
    df_neutral['confidence'] = 'auto'
    # ambiguous_df = df.loc[(df['auto_for'] & df['auto_against']) |
    #                       (~df['auto_for'] & ~df['auto_against'] & ~df['auto_neutral'])]

    # Remove the auto_* fields because they are useful only for computing the stance column value.
    df_adani.drop(columns=['auto_for', 'auto_against', 'auto_neutral'])

    # Save the auto-coded items in one file.
    autocoded_dataset_filepath = f'{coding_path}/auto_20100101-20180510.csv'
    logger.info(f'\tstoring auto-coded dataset file: {autocoded_dataset_filepath}')
    pd.concat([df_for, df_against, df_neutral]).to_csv(autocoded_dataset_filepath)
    print(f'\t\tfor: {get_size(df_for)}\n\t\tagainst: {get_size(df_against)} (sampled)\n\t\tneutral: {get_size(df_neutral)} (sampled)')


if __name__ == '__main__':
    fire.Fire(main)
    # Example invocation:
    # python autocoding_processor.py --dataset_filepath=/media/hdd_2/slo/stance/datasets/dataset.csv --coding_path=/media/hdd_2/slo/stance/coding
