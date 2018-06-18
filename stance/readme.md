# Stance Classifier

This stance classifier was added in early 2018, replacing the earlier
sentiment classifier(s).

`run_stance_detection.py` is the main script used to run the SLO
classifiers. The scripts, both Python and Bash, are known to run on OSx
and Linux, but some do not work on Windows (i.e.,
`data/train_wordvec.sh` and `tweet_preprocessor.py`).

The SLO Stance Datasets are stored separately in:
`Orca:/media/hdd_2/slo/`. See the readme file there for details on the
data.

## Data

The data sub-directory holds dataset processing tools that are
used for stance classification as follows:

0. Query tweets from ESL/Oracle in raw JSON form (see above).

1. `dataset_processor.py` - Builds a dataset file by:
    1. Reading a raw JSON file (e.g., `raw_tweets/dataset_20160101-20180304.json`).
    2. Dropping tweets by unknow companies or in non-English languages.
    3. Adding fields for language (TextBlob), full text, author screen
    name and profile description, hashtags.
    4. Writing a processed dataset file (e.g., `datasets/dataset.csv`) or
    set of company-specific dataset files (e.g., `datasets/dataset_adani.csv`).

2. The basic dataset files can be used to create coded train/testsets in
multiple ways:
    - *Manual coding* - `coding_processor.py` creates randomly sampled
    dataset files in CSV format for manual coding.
    - *Auto-coding* - `autocoding_processor.py` automatically codes sampled
    tweets based on hard-coded query terms.

3. Coded train/testsets can be preprocessed by `tweet_preprocessor.py`,
which tokenizes and normalizes the tweet and profile texts. Use these
 preprocessed datasets to:
    - *classification* - train and test classification models.
    - *Word Embeddings* -  create word-embedding vectors by:
        1. `extract_preprocessed.py` pulls out (only) the tokenized
        tweet texts.
        2. `train_wordvec.sh` runs FastText to produce word embeddings
        (e.g., `wordvec/adani_100.vec`).

## Models

The models sub-directory holds the classification models.

### Neural Networks

Three neural network (NN) models are implemented for this project, namely, CrossNet (`cn`), MemNet (`mn`), and Transformer (`tf`).
You can create a model instance by calling `model_factory.ModelFactory.get_model`.
All models can be called from `run_stance_detection.py` too.
Just change the model name (`--model`) to corresponding names of models.
Below is an example of conducting stance detection using CrossNet.

```shell
python run_stance_detection.py train --model=cn \
    --path=c:/projects/csiro/data/stance \
    --trainfp=coding/auto_20160101-20180510_tok.csv \
    --testfp=coding/gold_20180514_kvlinden_tok.csv \
    --wvfp=wordvec/20100101-20180510/all-100.vec \
    --noprofile \
    --logging_level=10
```

In addition, `run_stance_detection.py` allows you to specify arbitrary hyper parameters of NN models by json file.

```shell
python run_stance_detection.py train --model=cn \
    --path=c:/projects/csiro/data/stance \
    --trainfp=coding/auto_20160101-20180510_tok.csv \
    --testfp=coding/gold_20180514_kvlinden_tok.csv \
    --wvfp=wordvec/20100101-20180510/all-100.vec \
    --noprofile \
    --paramfp=models/example_params.json
```

The parameters specified by `--paramfp` will override the `--wvfp` and `--profile` settings.
In this way, you can try ad-hoc experiments with any hyper parameters you like.


#### Caveat

When you generated a model of them by calling `model_factory.ModelFactory.get_model`, all of the NN models are wrapped by `nn_utils.NeuralPipeline` so as to accept `sklearn` API.
Therefore, the model can be put to `sklearn`-style train/test codes.
However, there is one big different from usual `sklearn` routines, i.e., you must give a pair of the train input and the test input (like `x_train = [x_train, x_test]`) when you make the model `fit` to the training data.
See `run_stance_detection.train_pred` for the reference of an actual coding example.
This is because NN models requires the sequence of numerical indices corresponding to actual words as input (like `['hello', 'world'] -> [28, 137]`) in order to embed words into dense vectors (word embeddings).
Actually, the mapping between indices and words should be coherent between train examples and test examples (same words should have same indices regardless of the appearance in train/test).


## Tests

The tests sub-directory holds the unit tests.

## Tuning

### SVM

The hyper parameters of the SVM model is automatically tuned by grid search on each run.
You can specify candidates of hyper parameter values and even add other hyper parameters to consider by editing `models/svm_mohammad17.py`.

### Neural network models

NN models can be tuned by `run_grid_search.py` and/or `run_bayes_search.py`.
The former conducts hyper parameter search in a grid search manner or in a randomised search manner.
The latter executes it in a Bayesian search manner.
Both files have two ways to do each classification experiment, i.e., `fixed` and `xval`.
The `fixed` way corresponds to the setting where the model is always trained on a dataset for training and tested on a dataset for testing.
The `xval` way conducts cross validation where a given dataset is split to several parts and the model is trained and tested on different parts in rotation.
How to run the search is similar to `run_stance_detection.py`.

```
python run_grid_search fixed --model=crossnet --wvfp=data/wordvec/** \
    --trainfp=data/coding/** --testfp=data/coding/**

python run_grid_search xval --model=crossnet --wvfp=data/wordvec/** \
    --datafp=data/coding/**
```

See docstrings for details about other arguments and options.

The candidate hyper parameters to try are embedded in those files.
You can change the candidate values or ranges of hyper parameters by editing the files directly.
Before conducting the search, `git commit` is recommended to preserve the candidate information.

Grid search is a search strategy to try all of the hyper parameter combinations amongst finite candidates of each hyper parameter values.
This is a very common way to find a good hyper parameters but obviously takes a lot of time since the number of candidates explodes easily if the model has many parameters.
Randomised search is meant for solving the issue by searching best hyper parameters for randomly sampled candidate combinations.
Note that hyper parameters that take continuous values are sampled from probabilistic distributions you specify instead of finite candidate values as grid search tries.
Bayesian search is a method to sample hyper parameters more efficiently than just random, based on a Bayesian technique.
That is, after several executions of experiments, the method predicts a hyper parameter combination to try next which possibly produces better results, probabilistically judging from the results of past experiments.
Choose a precise method according to your computational environment.
