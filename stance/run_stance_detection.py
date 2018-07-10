"""Conduct stance detection experiments for each target.

This script loads a model for each target and applies it to the stance detection on the target.
This is meant for the template to code other experimental settings.
"""
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from fire import Fire

from data_utility import load_data, load_semeval_data
from keras.utils import to_categorical
from model_factory import ModelFactory
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def train_pred(model: Pipeline, modelname: str,
               x_train: np.ndarray, y_train: np.ndarray,
               x_test: np.ndarray, y_test: np.ndarray,
               target=None
               ) -> Tuple[float, float, np.ndarray, pd.DataFrame]:
    """Let model train (fit) and predict the data."""
    if modelname != 'svm':
        x_train = [x_train, x_test]
        # if model's loss == categorical_crossentropy
        y_train = to_categorical(y_train)
        # TODO: label smoothing
        # y_test = to_categorical(y_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    label_f = f1_score(y_test, y_pred, average=None)

    logger.debug(
        f'f1 for each label = {label_f}')
    logger.debug(f'confusion matrix\n{confusion_matrix(y_test, y_pred)}')
    macrof = f1_score(y_test, y_pred, labels=[0, 1, 2], average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    df_report = pd.DataFrame()
    df_report[f'{target}_against'], df_report[f'{target}_for'], df_report[f'{target}_neutral'] \
        = [pd.Series(label_f[0]), pd.Series(label_f[1]), pd.Series(label_f[2])]
    df_report[f'{target}_combined'] = pd.Series(macrof)

    return macrof, accuracy, y_pred, df_report


def report_result_ea(modelname: str, model: Pipeline,
                     target: str, macrof: float) -> None:
    if modelname == 'svm':
        logger.info(
            f'Target "{target}": {macrof:.4} macroF (param C: {model.best_params_["clf__C"]})')

        # print effective features
        names = model.best_estimator_.named_steps['vect'].get_feature_names()
        logger.debug(f'total feature size: {len(names)}')
        coef = model.best_estimator_.named_steps['clf'].coef_
        ranking = sorted(zip(names, coef[0]), key=lambda k: k[1])

        logger.debug(f'top 10 positive features:\n' + ' | '.join(
            ['{} ({:.3f})'.format(posf, v) for posf, v in reversed(ranking[-10:])]))
        logger.debug(f'top 10 negative features:\n' +
                     ' | '.join(['{} ({:.3f})'.format(negf, v) for negf, v in ranking[:10]]))
    else:
        logger.info(f'Target "{target}": {macrof:.4} macroF')


def run_semeval(modelname: str, datafp: str, wvfp: str,
                target: str='all', profile: bool=False
                ) -> None:
    """[WIP] conduct SemEval2016 TaskA.6 experiments.

    # Arguments
    datafp:
        Input data in which test and train are combined.
        If you use semeval dataset, it should contain 'Train' column which bears binary flags (True if the tweet belongs to the official train split).
        When you use SLO dataset, the experiment is conducted by k-fold cross validation.
    profile:
        use users' profile (description) texts for features. input file must contain 'profile_t' column
    """
    # x = Dict[target, ary_tweets], y = Dict[target, ary_labels]
    x_train_arys, y_train_arys, x_test_arys, y_test_arys = load_semeval_data(
        datafp, target=target, profile=profile)

    targets = x_train_arys.keys()
    logger.info('All targets: {}'.format(', '.join(targets)))
    y_pred_arys: Dict[str, np.ndarray] = {}
    fmacros: List[float] = []
    for t in targets:
        logger.info(f'Target: {t}')
        model = ModelFactory.get_model(
            modelname, wvfp, target=t, profile=profile)
        macrof, accuracy, y_pred, _ = train_pred(
            model, modelname,
            x_train_arys[t], y_train_arys[t],
            x_test_arys[t], y_test_arys[t]
        )
        report_result_ea(modelname, model, t, macrof)
        fmacros.append(macrof)
        y_pred_arys[t] = y_pred

    fmacro = np.mean(fmacros)
    logger.info(
        f'Overall: {fmacro:.4} macroF (in the **macro** mean over macroFs of all targets)')

    y_test_all = np.concatenate(list(y_test_arys.values()))
    y_pred_all = np.concatenate(list(y_pred_arys.values()))
    fmicro = f1_score(y_test_all, y_pred_all, labels=[0, 1], average='macro')
    logger.info(
        f'Overall: {fmicro:.4} macroF (in the __micro__ mean of results over all targets)')


def run_train(modelname,
              x_train_arys, y_train_arys,
              x_test_arys, y_test_arys,
              wvfp, profile, params=None):
    """This function runs a training epoch on the specified model on the given
    data. It will run all combinations of training and testing targets.
    """

    train_targets = x_train_arys.keys()
    test_targets = x_test_arys.keys()
    logger.info(f'train targets: {list(train_targets)}; '
                f'test targets: {list(test_targets)}')

    fmacros: List[float] = []
    accuracies: List[float] = []
    df_report = pd.DataFrame()

    # Train a model on each train target and test it on all test targets.
    for train_target in train_targets:

        model = ModelFactory.get_model(
            modelname, wvfp=wvfp,
            profile=profile, params=params)
        y_pred_arys: Dict[str, np.ndarray] = {}

        for test_target in test_targets:

            logger.info(f'train target: {train_target}; '
                        f'test target: {test_target}')
            macrof, accuracy, y_pred, df_new_columns = train_pred(
                model, modelname,
                x_train_arys[train_target], y_train_arys[train_target],
                x_test_arys[test_target], y_test_arys[test_target],
                target=test_target
            )

            for new_column in df_new_columns:
                df_report[new_column] = df_new_columns[new_column]

            report_result_ea(modelname, model, test_target, macrof)
            fmacros.append(macrof)
            accuracies.append(accuracy)
            y_pred_arys[test_target] = y_pred

    # Compute/print macro-f1 and micro-f1 summary statistics.
    fmacro = np.mean(fmacros)
    accuracy = np.mean(accuracies)
    logger.info(
        f'Overall: {fmacro:.4} macroF (in the **macro** mean over macroFs of all targets)')
    logger.info(
        f'Overall: {accuracy:.4} accuracy (in the accuracy mean over accuracies of all targets)')

    y_test_all = np.concatenate(list(y_test_arys.values()))
    y_pred_all = np.concatenate(list(y_pred_arys.values()))
    fmicro = f1_score(y_test_all, y_pred_all, labels=[
                      0, 1, 2], average='macro')
    logger.info(
        f'Overall: {fmicro:.4} microF (in the __micro__ mean of results over all targets)')

    return fmacro, df_report


def run_xval(modelname, x_arys, y_arys, wvfp,
             profile, cv=3, params=None):
    """Run a cross-validation on the specified model on the given data."""

    # TODO: implement micro Fmacro
    # y_pred_arys: Dict[str, np.ndarray] = {}
    fmacros4tgt: List[float] = []
    df_report = pd.DataFrame()

    for t in x_arys.keys():
        model = ModelFactory.get_model(
            modelname, wvfp,
            profile=profile, params=params)

        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        fmacros_: List[float] = []
        # y_pred_: List[np.ndarray] = []
        # y_test: List[np.ndarray] = []
        for i, (train_idx, test_idx) in enumerate(skf.split(x_arys[t], y_arys[t])):
            logger.debug(f'CV {i+1}')
            x_train = x_arys[t][train_idx]
            y_train = y_arys[t][train_idx]
            x_test = x_arys[t][test_idx]
            y_test_ea = y_arys[t][test_idx]
            macrof_ea, accuracy_ea, y_pred_ea, df_cv = train_pred(
                model, modelname,
                x_train, y_train,
                x_test, y_test_ea
            )
            df_report = df_report.append(df_cv)

            fmacros_.append(macrof_ea)
            # y_pred_.append(y_pred_ea)
            # y_test.append(y_test_ea)
            report_result_ea(modelname, model, t, macrof_ea)
        macrof = np.mean(fmacros_)
        macrof_std = np.std(fmacros_)
        # y_pred = np.concatenate(tuple(y_pred_))
        # y_test = np.concatenate(tuple(y_test))

        if modelname == 'svm':
            logger.info(
                f'Target "{t}": {macrof:.4} macroF (param C: {model.best_params_["clf__C"]})')
        else:
            logger.info(
                f'Target "{t}": {macrof:.4} +/- {macrof_std:.4} macroF (mean of {cv}CV)')
        fmacros4tgt.append(macrof)
        # y_pred_arys[t] = y_pred

    # fmacro = np.mean(fmacros4tgt)
    # logger.info(f'Over all targets: {fmacro:.4} macroF')

    # y_test_all = np.concatenate(list(y_test_arys.values()))
    # y_pred_all = np.concatenate(list(y_pred_arys.values()))
    # fmicro = f1_score(y_test_all, y_pred_all, labels=[0, 1], average='macro')
    # logger.info(f'Overall: {fmicro:.4} macroF (in the __micro__ mean of results over all targets)')

    return fmacros4tgt, df_report


class Interface:
    def __init__(self,
                 model='svm',
                 path='.',
                 wvfp='all_100.vec',
                 target='all',
                 profile=True,
                 paramfp=None,
                 repeat=1, cv=3,
                 logging_level=logging.INFO,
                 logging_filename=None,
                 ):
        """Initialize stance detection system.

        Keyword Arguments
        :param model: the name of the learning model to use (svm, crossnet, ...)
            (default: svm)
        :param path: the root path for all filepaths
            (default: '.')
        :param wvfp: the system filepath of the word embedding vectors. when you specify '--nowvfp', no pre-trained word embeddings are used
            (default: 'wordvec/all_100.vec')
        :param target: the company to use as the stance target
                (default='all')
        :param profile: use the users' profile description texts for features;
                the input file must contain a 'profile_t' column.
                (default=True)
        :param paramfp: the file path to the parameter configuration file for neuran network models. JSON format is acceptable.
                (default=None)
        :param repeat: the number of times to run the classifier, only for 'train'
                (default=1)
        :param cv: the number of cross validation split, only for 'xval'
                (default=3)
        :param logging_level: the level of logging to use (DEBUG - 10; INFO - 20; ...)
            (default: logging.INFO)
        :param logging_filename: the file in which to save logging output, if any
            (default: None)
        """
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=logging_filename,
        )
        self.model = model
        self.path = path
        self.wvfp = os.path.join(self.path, wvfp) if wvfp else None
        self.target = target
        self.profile = profile
        if paramfp:
            with open(paramfp, 'r') as f:
                self.params = json.load(f)
            if 'profile' in self.params:
                # override `profile` value with paramfp's one
                self.profile = self.params.pop('profile')
        else:
            self.params = None
        self.repeat = repeat
        self.cv = cv
        self.logging_level = logging_level

    def semeval(self, modelname: str, datafp: str, wvfp: str,
                target: str='all', profile: bool=False
                ) -> None:
        # FIXME: API outdated
        if modelname != 'svm':
            # iterate several times
            for i in range(1, 21):
                logger.info(f'==== Experiment {i} ====')
                run_semeval(modelname, datafp, wvfp,
                            target=target, profile=profile)
        else:
            run_semeval(modelname, datafp, wvfp,
                        target=target, profile=profile)

    def train(self, trainfp, testfp, outfp=None):
        """Run a standard train/test cycle on the given data.

        Keyword Arguments
        :param trainfp: the system filepath of the CSV training dataset
        :param testfp: the system filepath of the CSV testing dataset
        :return: macro-F score average and std-dev
        """
        logger.info(f'training {self.model} for {self.repeat} iterations')
        trainfp = os.path.join(self.path, trainfp)
        testfp = os.path.join(self.path, testfp)
        fmacro_list = []
        x_train_arys, y_train_arys = \
            load_data(trainfp, target=self.target, profile=self.profile)
        x_test_arys, y_test_arys = \
            load_data(testfp, target=self.target, profile=self.profile)
        for i in range(self.repeat):
            if self.repeat > 1:
                logger.info(f'iteration: {i+1}')
            fmacro, df_report = run_train(self.model,
                               x_train_arys, y_train_arys,
                               x_test_arys, y_test_arys,
                               self.wvfp, self.profile,
                               params=self.params)
            fmacro_list.append(fmacro)
        average = np.average(fmacro_list)
        stdev = np.std(fmacro_list)
        logger.info(f'total iterations: {self.repeat}; '
                    f'fmacro average: {average:.4}; '
                    f'fmacro std dev: {stdev:.4}'
                    )

        df_report['combined'] = pd.Series(average)
        if outfp:
            with open(outfp, "a") as f:
                df_report.to_csv(f, header=False, index=False)

        return average, stdev

    def xval(self, datafp, outfp):
        """Run a cross-validation test on the given data.

        Keyword Arguments
        :param datafp: the system filepath of the CSV dataset
        :return: macro-F score average and std-dev
        """
        logger.info(f'running {self.model} X-val for {self.repeat} iterations')
        datafp = os.path.join(self.path, datafp)
        x_arys, y_arys = load_data(
            datafp, target=self.target, profile=self.profile)
        fmacro_list, df_report = run_xval(self.model, x_arys, y_arys,
                               self.wvfp, self.profile,
                               cv=self.cv, params=self.params)
        average = np.average(fmacro_list)
        stdev = np.std(fmacro_list)
        logger.info(f'total CV iterations: {self.cv}; '
                    f'target-wise average: {average:.4}; '
                    f'target-wise std dev: {stdev:.4}')

        if outfp:
            df_mean = pd.DataFrame(df_report.mean()).transpose()
            with open(outfp, "a") as f:
                df_mean.to_csv(f, header=False, index=False)

        return average, stdev


if __name__ == '__main__':
    Fire(Interface)
    # Example invocation - training:
    # python run_stance_detection.py train --model=svm --path=c:/projects/csiro/data/stance --trainfp=coding/auto_20160101-20180510_tok.csv --testfp=coding/gold_20180514_kvlinden_tok.csv --wvfp=wordvec/20100101-20180510/all-100.vec --logging_level=10
    # Example invocation - xval:
    # python run_stance_detection.py xval --model=svm --path=c:/projects/csiro/data/stance --datafp=coding/coding_2018-05-02_auto_tok.csv --wvfp=wordvec/all_100.vec
