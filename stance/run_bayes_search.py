
"""Grid Search (especially for neural models)."""
import logging
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
from fire import Fire
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from data_utility import load_data
from run_stance_detection import run_fixed_fast, run_train, run_xval

logger = logging.getLogger(__name__)


def get_param_dists(modelname):
    """Generate distributions of hyper parameter candidates

    Directly edit the candidate parameter ranges here.
    Recommend to make `git commit` before running the script in order to record what you did.
    """
    param_combs = dict(
        # max_vocabsize=hp.choice('max_vocabsize', [100_000, 200_000, 300_000]),
        max_vocabsize=10000,
        # max_tgtlen=[1, 4],
        max_tgtlen=1,  # for SLO, we just need 1 token
        dim_wordvec=hp.choice('dim_wordvec', [64, 128, 256, 512]),
        prf_cat=hp.choice('prf_cat', [False, True]),
        # profile=hp.choice('profile', [False, True]),
        profile=True,
        max_prflen=hp.choice('max_prflen', [20, 40]),
        dropout=hp.uniform('dropout', 0.1, 0.5),  # an advice from Shaukat
        lr=hp.loguniform('lr', -5, -1),
        validation_split=0.2,
        epochs=200,  # note that early_stopping is applied
        batch_size=hp.choice('batch_size', [32, 64, 128, 256]),
        patience=30  # early stopping
    )
    if modelname in ['crossnet', 'cn', 'CrossNet', 'crossNet']:
        param_combs['dim_lstm'] = hp.choice(
            'dim_lstm', [100, 200, 300, 400, 500])
        # param_combs['num_reason'] = hp.choice('num_reason', [1, 2, 3])
        param_combs['num_reason'] = 1
        param_combs['dim_dense'] = hp.choice('dim_dense', [100, 200, 300])
    elif modelname in ['memnet', 'MemNet', 'mn', 'memNet', 'AttNet', 'attnet']:
        param_combs['dim_lstm'] = hp.choice(
            'dim_lstm', [100, 200, 300, 400, 500])
        param_combs['num_layers'] = hp.choice('num_layers', [1, 2, 3, 4])
        param_combs['weight_tying'] = hp.choice('weight_tying', [False, True])
    elif modelname in ['tf', 'transformer', 'Transformer']:
        param_combs['m_profile'] = hp.choice('m_profile', [1, 2])
        # param_combs['target'] = hp.choice('target', [False, 1, 2])
        param_combs['target'] = 1
        # param_combs['parallel'] = hp.choice('parallel', [False, 1, 2, 3])
        param_combs['parallel'] = 1
        param_combs['xtra_self_att'] = hp.choice(
            'xtra_self_att', [False, True])
        # param_combs['dim_pff'] = hp.choice('dim_pff', [64, 128, 256, 512])  # dynamic
        param_combs['num_head'] = hp.choice('num_head', [2, 4, 8])
        param_combs['num_layers'] = hp.choice('num_layers', [1, 2, 3, 4])
    else:
        raise NotImplementedError

    return param_combs


class BayesianSearch():
    """The main interface to conduct Bayesian search.

    Using `hyperopt` package, this script conducts Bayesian hyper parameter tuning with TPE algorithm.
    if `trainfp` is not None, search is conducted on the fixed train/test split.
    else, 3-fold cross validation on `datafp` is executed.
    """

    def __init__(self, model, wvfp,
                 evals=100,
                 repeat=3, cv=3,
                 path='.',
                 logging_level=logging.DEBUG,
                 logging_filename=None
                 ):
        """Initialize stance detection system.

        Keyword Arguments
        :param model: the name of the learning model to use (svm, crossnet, ...)
        :param wvfp: the system filepath of the word embedding vectors
        :param evals: the number of hyper parameter combinations to try (default: 100)
        :param repeat: the number of times to run the classifier per each parameter combination
                (default: 3)
        :param cv: the number of cross validation split, only for 'xval'
                (default=3)
        :param path: the root path for all filepaths
            (default: '.')
        :param logging_level: the level of logging to use (DEBUG - 10; INFO - 20; ...)
            (default: logging.DEBUG)
        :param logging_filename: the file in which to save logging output, if any
            (default: grid_search-MODEL-DATE.log)
        """
        if model == 'svm':
            raise ValueError(
                'This script is currently incompatible with SVM. FYI, SVM is always tuned when they fit to data. (see models.svm_mohammad17)')
        self.model = model
        self.path = path
        self.wvfp = os.path.join(self.path, wvfp) if wvfp else None
        self.evals = evals
        self.repeat = repeat
        self.cv = cv

        # logger setup
        self.startdate = datetime.now()
        self.basename = f'bayes_search-{self.model}-{self.startdate:%d%b%Y}'
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=logging_filename if logging_filename else self.basename + '.log',
        )
        # simultaenuously output INFO logs to stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger(__name__).addHandler(console)

        # TODO: need wordvec routine
        # pass a folder path containing wordvecs
        # load once for each wv file and pass those objects (KeyedVectors) as parameters

    def fixed(self, trainfp, testfp):
        """Run a standard train/test cycle on the given data.

        Keyword Arguments
        :param trainfp: the system filepath of the CSV training dataset
        :param testfp: the system filepath of the CSV testing dataset
        """
        logger.info(
            f'BAYESIAN SEARCH on the model "{self.model}" '
            f'for {self.evals} parameter combinations '
            f'with {self.repeat} repetition')

        trainfp = os.path.join(self.path, trainfp)
        testfp = os.path.join(self.path, testfp)
        self.x_train_arys, self.y_train_arys = \
            load_data(trainfp, target='all', profile=False)
        self.x_train_arys_p, self.y_train_arys_p = \
            load_data(trainfp, target='all', profile=True)
        self.x_test_arys, self.y_test_arys = \
            load_data(testfp, target='all', profile=False)
        self.x_test_arys_p, self.y_test_arys_p = \
            load_data(testfp, target='all', profile=True)

        trials = Trials()
        best = fmin(
            self._run_fixed,
            space=get_param_dists(self.model),
            algo=tpe.suggest,
            max_evals=self.evals,
            trials=trials,
            verbose=1
        )

        with open(f'{self.basename}.pkl', 'wb') as f:
            pickle.dump(trials, f)
        logger.info(f'the best parameters are: {best}')
        logger.info(f'the f1 of which is: {trials.best_trial["result"]}')

    def xval(self, datafp):
        """Run a cross-validation test on the given data.

        Keyword Arguments
        :param datafp: the system filepath of the CSV dataset
        """
        if self.rand:
            searchname = f'RANDOM SEARCH on the model "{self.model}" '
        else:
            searchname = f'Exhaustive GRID SEARCH on the model "{self.model}" '
        logger.info(
            searchname +
            f'for {len(self.param_grid)} parameter combinations '
            f'with {self.repeat} repetition')
        datafp = os.path.join(self.path, datafp)
        self.x_arys, self.y_arys = load_data(
            datafp, target='all', profile=False)
        self.x_arys_p, self.y_arys_p = load_data(
            datafp, target='all', profile=True)

        trials = Trials()
        best = fmin(
            self._run_xval,
            space=get_param_dists(self.model),
            algo=tpe.suggest,
            max_evals=self.evals,
            trials=trials,
            verbose=1
        )

        with open(f'{self.basename}.pkl', 'wb') as f:
            pickle.dump(trials, f)
        logger.info(f'the best parameters are: {best}')
        logger.info(f'the f1 of which is: {trials.best_trial["result"]}')

    def _run_fixed(self, params):
        logger.info(f'params: {params}')
        profile = params.pop('profile', None)
        fmacro_list: List[float] = []
        for i in range(self.repeat):
            if self.repeat > 1:
                logger.info(f'iteration: {i+1}')
            if profile:
                fmacro = run_fixed_fast(self.model,
                                        self.x_train_arys_p, self.y_train_arys_p,
                                        self.x_test_arys_p, self.y_test_arys_p,
                                        self.wvfp, profile,
                                        params=params)
            else:
                fmacro = run_fixed_fast(self.model,
                                        self.x_train_arys, self.y_train_arys,
                                        self.x_test_arys, self.y_test_arys,
                                        self.wvfp, profile,
                                        params=params)
            fmacro_list.append(fmacro)
        avgfmacro = np.mean(fmacro_list)
        varfmacro = np.var(fmacro_list)
        # TODO: shall we use 'true_loss' and 'true_loss_variance' too?
        return {'loss': -1 * avgfmacro, 'status': STATUS_OK, 'loss_variance': varfmacro}

    def _run_xval(self, params):
        logger.info(f'params: {params}')
        profile = params.pop('profile', None)
        if profile:
            fmacro_list = run_xval(self.model, self.x_arys_p, self.y_arys_p,
                                   self.wvfp, profile,
                                   cv=self.cv, params=params)
        else:
            fmacro_list = run_xval(self.model, self.x_arys, self.y_arys,
                                   self.wvfp, profile,
                                   cv=self.cv, params=params)
        avgfmacro = np.mean(fmacro_list)
        varfmacro = np.var(fmacro_list)
        # TODO: shall we use 'true_loss' and 'true_loss_variance' too?
        return {'loss': -1 * avgfmacro, 'status': STATUS_OK, 'loss_variance': varfmacro}


if __name__ == '__main__':
    Fire(BayesianSearch)
