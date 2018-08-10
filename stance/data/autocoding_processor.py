import datetime
import fire
import re

import pandas as pd

import logging
logger = logging.getLogger(__name__)

from data.settings import PTN_against, PTN_for, PTN_neutral_screennames, PTN_company_usernames, company_list


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

    pd.options.mode.chained_assignment = None  # default='warn'

    if testset_filepath:
        logger.info(f'\tloading testset file: {testset_filepath}')
        df_testset = pd.read_csv(testset_filepath, encoding=encoding, engine='python')
        test_ids = re.compile('|'.join(pd.Series(df_testset['id']).apply(str)))

    logger.info(f'\tloading dataset file: {dataset_filepath}')
    df_all = pd.read_csv(dataset_filepath, encoding=encoding, engine='python')
    logger.info(f'\t\t{get_size(df_all)} items loaded')

    df_combined = pd.DataFrame()

    for company in company_list:
        if testset_filepath:
            df_companies = df_all.loc[(df_all['company'].str.lower().str.contains(company)) & (~df_all['id'].astype(str).str.match(test_ids))]
        else:
            df_companies = df_all.loc[(df_all['company'].str.lower().str.contains(company))]
        logger.info(f'\t\t{get_size(df_companies)} {company} items loaded')

        # Replace all semicolons to fix column displacement issues.
        df_companies['text'] = df_companies['text'].str.replace(";", "")
        df_companies['user_description'] = df_companies['user_description'].str.replace(";", "")

        # Annotate tweets with suspected stance values using rule patterns.
        if companytweets:
            df_companies['auto_for'] = (df_companies['text'].str.lower().str.contains(PTN_for[company]) |
                                        df_companies['user_screen_name'].str.lower().str.match(PTN_company_usernames))
        else:
            df_companies['auto_for'] = df_companies['text'].str.lower().str.contains(PTN_for[company])
        df_companies['auto_against'] = df_companies['text'].str.lower().str.contains(PTN_against[company])
        df_companies['auto_neutral'] = df_companies['user_screen_name'].str.lower().str.match(PTN_neutral_screennames)

        # Collect tweets that are to be coded for each stance value.
        df_for = df_companies.loc[df_companies['auto_for'] & ~df_companies['auto_against']
                                  & ~df_companies['auto_neutral'] & ~df_companies['retweeted']]
        df_against = df_companies.loc[~df_companies['auto_for'] & df_companies['auto_against']
                                      & ~df_companies['auto_neutral'] & ~df_companies['retweeted']]
        df_neutral = df_companies.loc[~df_companies['auto_for'] & ~df_companies['auto_against']
                                      & df_companies['auto_neutral'] & ~df_companies['retweeted']]

        s_min = min(df_for.shape[0], df_against.shape[0], df_neutral.shape[0])

        # Get sample of for tweets
        for_max_size = df_for.shape[0]
        if for_sample_size and for_sample_size <= for_max_size:
            df_for = df_for.sample(for_sample_size)
        elif for_sample_size:
            logger.debug("\nNot enough 'for' tweets for specified sample size\n")
            exit(-1)
        else:
            if s_min > 0:
                df_for = df_for.sample(s_min)
        df_for['stance'] = 'for'
        df_for['confidence'] = 'auto'
        df_for['company'] = company

        # Get sample of against tweets
        against_max_size = df_against.shape[0]
        if against_sample_size and against_sample_size <= against_max_size:
            df_against = df_against.sample(against_sample_size)
        elif against_sample_size:
            logger.debug("\nNot enough 'against' tweets for specified sample size\n")
            exit(-1)
        else:
            if s_min > 0:
                df_against = df_against.sample(s_min)
        df_against['stance'] = 'against'
        df_against['confidence'] = 'auto'
        df_against['company'] = company

        # Get sample of neutral tweets
        neutral_max_size = df_neutral.shape[0]
        if neutral_sample_size and neutral_sample_size <= neutral_max_size:
            df_neutral = df_neutral.sample(neutral_sample_size)
        elif neutral_sample_size:
            logger.debug("\nNot enough 'neutral' tweets neutral specified sample size\n")
            exit(-1)
        else:
            if s_min > 0:
                df_neutral = df_neutral.sample(s_min)
        df_neutral['stance'] = 'neutral'
        df_neutral['confidence'] = 'auto'
        df_neutral['company'] = company

        # ambiguous_df = df.loc[(df['auto_for'] & df['auto_against']) |
        #                       (~df['auto_for'] & ~df['auto_against'] & ~df['auto_neutral'])]

        # Remove the auto_* fields because they are useful only for computing the stance column value.
        df_companies.drop(columns=['auto_for', 'auto_against', 'auto_neutral'])

        df_combined = df_combined.append(pd.concat([df_for, df_against, df_neutral]))
        print(f'\tCompany: {company}\n\t\tfor: {get_size(df_for)} (out of {for_max_size})\n\t\tagainst: '
              f'{get_size(df_against)} (out of {against_max_size})\n\t\tneutral: {get_size(df_neutral)} '
              f'(out of {neutral_max_size})')
        index = coding_filepath.find('.csv')
        company_coding_filepath = coding_filepath[:index] + "_" + company + coding_filepath[index:]
        pd.concat([df_for, df_against, df_neutral]).to_csv(company_coding_filepath)
    # Save the auto-coded items in one file.
    autocoded_dataset_filepath = f'{coding_filepath}'
    logger.info(f'\tstoring auto-coded dataset file: {autocoded_dataset_filepath}')
    df_combined.to_csv(autocoded_dataset_filepath)


if __name__ == '__main__':
    fire.Fire(main)
    # Example invocation:
    # python autocoding_processor.py --dataset_filepath=/media/hdd_2/slo/stance/datasets/dataset.csv --coding_filepath=/media/hdd_2/slo/stance/coding --against_muliplier=1 --for_multiplier=1
