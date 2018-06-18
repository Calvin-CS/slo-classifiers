"""Utility modules of data handling on machine learning.

borrow some functions from github.com/nuaaxc/cross_target_stance_classification
"""
import csv
from typing import Dict, List, Tuple

import numpy as np

from data.settings import PTN_against_hashtags, PTN_for_hashtags

import logging
logger = logging.getLogger(__name__)

Dsets = Dict[str, np.ndarray]


def dic_list2ary(lists: Dict[str, list]) -> Dsets:
    return {target: np.array(lst) for target, lst in lists.items()}


def get_x(row: dict, rm_autotag: bool, profile: bool) -> str:
    """Utility function to add metadata along with main tweet text."""

    # The output is the target company, tweet and, if requested, profile.
    # The order of the features must match the order used in split_x_value().
    features = list()
    features.append(row['company'])
    features.append(row['tweet_t'])
    if profile:
        features.append(row['profile_t'])
    output = '\t'.join(features)

    # Remove auto-tagging hashtags, if requested, from the tweet and profile.
    if rm_autotag:
        output = PTN_for_hashtags.sub('', output)
        output = PTN_against_hashtags.sub('', output)

    return output
    # add target information for CrossNet
    # return '\t'.join([
    #     row['tweet_t'],
    #     row['target'].lower()
    # ])


def split_x_value(x, profile_flag):
    """This function returns the company target and tweet text strings. If
    profile is true, include the profile text as well, otherwise return None
    as the profile text. The order of the split must match the order in get_x();
    the order of the return values is: target, tweet, profile.
    """
    if profile_flag:
        target, tweet, profile = x.split('\t')
    else:
        target, tweet = x.split('\t')
        profile = ''
    return target, tweet, profile


def load_semeval_data(datafp: str, target: str, profile: bool) -> Tuple[Dsets, Dsets, Dsets, Dsets]:
    """Load SemEval2016 Task6.A datasets for the specified target.

    # Args
    datafp:
        Input file path. The file must contain 'tweet_t' and 'Train' columns.
    target:
        'all' for all targets.
        You can specify one specific target by its acronym.
        Valid instances are follows.
        # TODO: list and implement acronyms

    # Return
    Dict of tweet/label arrays in the following order.
    - Train tweets
    - Train labels
    - Test tweets
    - Test labels
    """

    x_train_lists: Dict[str, List[str]] = {}
    y_train_lists: Dict[str, List[int]] = {}
    x_test_lists: Dict[str, List[str]] = {}
    y_test_lists: Dict[str, List[int]] = {}

    # NOTE: should be negative = 0, positive = 1 due to the calculation for macroF measure
    labels = ['AGAINST', 'FAVOR', 'NONE']
    logger.info('SemEval labels ' + ', '.join([f'{i}:{l}' for i, l in enumerate(labels)]))

    logger.info(f'loading data {datafp}')
    with open(datafp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row['Target']
            if target != 'all' and t != target:
                continue

            x = get_x(row, profile=profile)  # must contain preproc-ed tweets
            y = labels.index(row['Stance'].strip())

            if row['Train'].lower() == 'true':  # still string!
                x_train_lists.setdefault(t, []).append(x)
                y_train_lists.setdefault(t, []).append(y)
            else:
                x_test_lists.setdefault(t, []).append(x)
                y_test_lists.setdefault(t, []).append(y)

    x_train_arys = dic_list2ary(x_train_lists)
    y_train_arys = dic_list2ary(y_train_lists)
    x_test_arys = dic_list2ary(x_test_lists)
    y_test_arys = dic_list2ary(y_test_lists)

    logger.info('data loaded.')

    return x_train_arys, y_train_arys, x_test_arys, y_test_arys


def load_data(datafp: str, target: str, profile: bool) -> Tuple[Dsets, Dsets]:
    """Load SLO datasets for a specified target.

    Mostly same as `load_semeval_data` but doesn't split train/test sets.
    """
    logger.info(f'loading data from {datafp} for target {target}')

    x_lists: Dict[str, List[str]] = {}
    y_lists: Dict[str, List[int]] = {}

    # NOTE: should be negative = 0, positive = 1 due to the calculation for macroF measure
    labels = ['against', 'for', 'neutral', 'na']
    logger.debug('labels ' + ', '.join([f'{i}:{l}' for i, l in enumerate(labels)]))

    # TODO: a securer way might needed
    # detect whether input data are auto-coded or not
    rm_autotag = 'auto' in datafp
    if rm_autotag:
        logger.info('detected auto-coded data - removing query hashtags from tweet texts...')

    with open(datafp, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row['company']
            if target != 'all' and t != target:
                continue

            x = get_x(row, rm_autotag=rm_autotag, profile=profile)
            y = labels.index(row['stance'].strip())

            x_lists.setdefault(t, []).append(x)
            y_lists.setdefault(t, []).append(y)

    x_arys = dic_list2ary(x_lists)
    y_arys = dic_list2ary(y_lists)

    logger.debug('size per company: ' + ', '.join([f'{k}: {v.shape[0]}' for k, v in x_arys.items()]))

    return x_arys, y_arys


def load_combined_data(datafp, labels, profile=True):
    """Load a SLO dataset and return X and Y.

    Mostly same as `load_data` but doesn't split the target datasets and
    doesn't remove query hashtags.
    """
    logger.info(f'loading combined dataset from {datafp}')

    x_items = []
    y_items = []

    with open(datafp, encoding='utf-8') as f:

        for row in csv.DictReader(f):
            x = get_x(row, rm_autotag=False, profile=profile)
            y = labels.index(row['stance'].strip())

            x_items.append(x)
            y_items.append(y)

    return np.asarray(x_items), np.asarray(y_items)


