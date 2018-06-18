"""Ad-hoc script for extracting preprocessed tweet text from datasets.

Please execute ../tweet_preprocessor.py to dataset files and obtain dataset_*_tok.csv in advance.
This script should be executed at under stance/data as default.
"""
import logging
import os
import pandas as pd
from fire import Fire

logger = logging.getLogger(__name__)


def extract_preprocessed(root_path='.',
                         dataset_path='datasets',
                         dataset_filename='dataset_tok.csv',
                         output_path='preprocessed_tweets/slo',
                         encoding='utf-8',
                         logging_level=logging.INFO):
    """This function extracts the raw text from the given tokenized dataset
    file. The text includes both the tweet text and the user profile
    description text. The input is assumed to have been tokenized by
    tweet_preprocessor.py. The output is written to path/filename.txt.

    Keyword Arguments
        :param root_path: the root system path to the target/destination files
            (default: .)
        :param dataset_path: the name of the dataset sub-directory
            (default: datasets)
        :param dataset_filename: the name of the dataset file
            (default: dataset_tok.csv)
        :param output_path: the sub-directory into which to put the results
            (default: preprocessed_tweets)
        :param encoding: the file encoding
            (default: utf-8)
        logging_level -- the level of logging to use
            (default: logging.INFO)
    """
    logging.basicConfig(level=logging_level, format='%(message)s')

    dataset_filepath = f'{root_path}/{dataset_path}/{dataset_filename}'
    if not os.path.isfile(dataset_filepath):
        logger.fatal(f'\tdataset file doesn\'t exist: {dataset_filepath}')
        exit(-1)
    if not dataset_filename.endswith('_tok.csv'):
        logger.fatal(f'\tdataset isn\'t properly tokenized: {dataset_filepath}')
        exit(-1)

    logger.info(f'loading {dataset_filepath}...')
    df = pd.read_csv(dataset_filepath)
    output_filename = dataset_filename.split('.')[0]
    output_filepath = f'{root_path}/{output_path}/{output_filename}.txt'
    users = set()
    with open(output_filepath, 'w', encoding=encoding) as fout:
        logger.info(f'writing to {output_filepath}...')
        # Dump unique tweet and profile texts (separately).
        fout.writelines([text for text in df['tweet_t'].unique() + '\n'])
        fout.writelines([text for text in df['profile_t'].unique() + '\n'])
    # Older code for SemEval testing...
    # elif dataset_path == 'SemEval2016taskA':
    #     # join testsplit and trainsplit
    #     outp = f'preprocessed_tweets/semeval/{name}.txt'
    #     with open(outp, 'a') as f:
    #         f.writelines([tweet + '\n' for tweet in df['tweet_t']])
    #         print('\t->', outp, 'with appending')


if __name__ == '__main__':
    Fire(extract_preprocessed)
    # Example invocation:
    # python extract_preprocessed.py --path=/media/hdd_2/slo/stance --dataset_filename=dataset_20100101-20180510_tok.csv
