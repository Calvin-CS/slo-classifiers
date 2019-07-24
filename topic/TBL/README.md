# Social License to Operate Triple-Bottom-Line Topic Classification Project

&nbsp;

## Project Vision:

The purpose of this research project is to assist in the evaluation of the Social License to Operate
for organizations and other entities.  Eventually, we hope to be able to provide machine learning trained models
that can predict with relative accuracy how accepted a project is in the eyes of the public and stakeholders via
textual data obtained from social media outlets like Twitter.

&nbsp;

## Project Directory Structure:

"Data" - SLO Twitter Labeled Dataset Processing and Analysis (currently empty).

"Model" - SLO Twitter Labeled Dataset Topic Classification Algorithms.

&nbsp;

### Models Directory:

- "images" directory - stores .png and .jpeg files used in Jupyter Notebook files.

- "notebooks" directory - stores Jupyter Notebooks files.

- "slo_tbl_dataset_preprocessor.py" - pre-processes the labeled dataset in preparation for TBL topic classification.

- "slo_topic_classification_models.py" - contains functions implementing each algorithm for TBL topic classification.

- "slo_tbl_topic_classification_grid_searches.py" - contains functions implementing exhaustive grid search for each
algorithm for TBL topic classification.

&nbsp;

#### Jupyter Notebook Files in "notebooks" sub-directory:

report.ipynb: Contains project discussions including vision statements, background information, code implementation,
research results, and research implications.

showcase(demo).ipynb: Demos our machine learning implementation using the Multinomial Naive Bayes Classifier.

showcase(demo2).ipynb: Same as above, except shows overall implementation using all Classifiers.

showcase(metrics).ipynb: Contains metrics for side-by-side comparison of all Classifiers used as well as some
information on the datasets being used.

&nbsp;

## Codebase Execution:

### Tweet Pre-processing:

Run "slo_tbl_dataset_preprocessor.py"

### Description:

The Tweet preprocessor contains 3 separate functions that each pre-processes a specific dataset.
Comment/Uncomment the associated functions in the driver program "main" to pre-process the datasets.

Note: The import of the required datasets assumes a relative path from the root project directory of:

- /tbl-datasets/
- /borg-SLO-classifiers/

Note: Datasets are not included in the repository.

- dataset_20100101-20180510_tok.csv (obtained from Borg supercomputer)
- tbl_training_set.csv (obtained from Professor VanderLinden)
- tbl_kvlinden.csv (obtained from Professor VanderLinden)

&nbsp;

### Important Notes:

Please note the code snippets below:

```python
# Note: Need to set level AND turn on debug variables in order to see all debug output.
log.basicConfig(level=log.DEBUG)

# Miscellaneous parameter adjustments for pandas and python.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Turn on and off to debug various sub-sections.
debug = True
```

```python
"""
Main function.  Execute the program.

Note: Used to individually test that the preprocessors function as intended.
"""

if __name__ == '__main__':

    start_time = time.time()

    """
    Comment or uncomment in order to run the associated tweet preprocessor module.
    """
    # tweet_dataset_preprocessor_1()
    # tweet_dataset_preprocessor_2()
    # tweet_dataset_preprocessor_3()

    end_time = time.time()

    if debug:
        log.debug("\n")
        log.debug("Time taken to run pre-processor function:")
        time_taken = end_time - start_time
        log.debug(time_taken)
        log.debug("\n")
```

&nbsp;

### Tweet Classification:

Run "slo_topic_classification_models.py"

### Description:

The Tweet topic classification program is currently one large monolithic codebase that sequentially
trains, predicts, and visualizes the results of various Scikit-Classifiers.
None of these Classifiers are neural networks except the Multi-Layer Perceptron Classifier.

Note: The import of the required datasets assumes a relative path from the root project directory of:

- /preprocessed-datasets/

&nbsp;

The Python code will automatically import the datasets and prep them to be used as a input feature and target label set.
Then, it will run various Classifiers and output metric summaries and visualizations for each one to the console output.
You can control which Classifier is run by commenting/uncommenting the relevant function calls in the "main" driver
program at the very bottom of the codebase!

&nbsp;

TODO - tune hyper parameters for better training results and enhanced predictive ability.

TODO - clean and/or refactor the code base for style and readability purposes.

&nbsp;

### Important Notes:

Please note the comments in this code snippet below:

```python
# Note: Need to set level AND turn on debug variables in order to see all debug output.
log.basicConfig(level=log.DEBUG)
tf.logging.set_verbosity(tf.logging.ERROR)

# Miscellaneous parameter adjustments for pandas and python.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

"""
Turn debug log statements for various sections of code on/off.
"""
# Debug the scikit_learn_multinomialnb_classifier_non_pipeline() function.
debug_MNB_nonPipeline = False
# Debug the GridSearch functions for each Classifier.
debug_pipeline = False
# Debug the initial dataset import and feature/target set creation.
debug_preprocess_tweets = False
# Debug create_training_and_test_set() function.
debug_train_test_set_creation = False
# Debug each iteration of training and predictions for each Classifier function().
debug_classifier_iterations = True
# Debug the create_prediction_set() function.
debug_create_prediction_set = False
# Debug the make_predictions() function.
debug_make_predictions = False

"""
Controls the # of iterations to run each Classifier before outputting the mean accuracy metric obtained.

IMPORTANT NOTE: SET to "1" unless you want each Classifier spitting out visualizations for each of N iterations
when visualizations are enabled!
"""
iterations = 1

"""
Enable or disable making predictions using trained model on our large 650k+ Tweet dataset (takes a little while).
"""
enable_predictions = False

"""
Enable or disable plotting graph visualizations of the model's training and prediction results.
"""
enable_visualizations = True
```

&nbsp;

Please note the comments in this code snippet below:

```python
"""
Main function.  Execute the program.
"""
import time

if __name__ == '__main__':

    # For debug purposes.
    # my_set = create_prediction_set()

    start_time = time.time()

    # Call non-pipelined multinomial Naive Bayes Classifier training function.
    # scikit_learn_multinomialnb_classifier_non_pipeline()

    ################################################
    """
    This section calls grid search functions for automated hyper parameter tuning.
    """
    # multinomial_naive_bayes_classifier_grid_search()
    # sgd_classifier_grid_search()
    # svm_support_vector_classification_grid_search()
    # svm_linear_support_vector_classification_grid_search()
    # nearest_kneighbor_classifier_grid_search()
    # decision_tree_classifier_grid_search()
    # multi_layer_perceptron_classifier_grid_search()
    # logistic_regression_classifier_grid_search()

    ################################################
    """
    This section calls Scikit-Learn classifer functions for model training and prediction.
    """
    multinomial_naive_bayes_classifier()
    sgd_classifier()
    svm_support_vector_classification()
    svm_linear_support_vector_classification()
    nearest_kneighbor_classifier()
    decision_tree_classifier()
    multi_layer_perceptron_classifier()
    logistic_regression_classifier()

    ################################################

    end_time = time.time()

    if debug_pipeline:
        log.debug("The time taken to train the classifier(s), make predictions, and visualize the results is:")
        total_time = end_time - start_time
        log.debug(str(total_time))
        log.debug("\n")
```

&nbsp;

EndNote: For specific details concerning our code implementation refer to the Python files.
Otherwise, refer to report.ipynb "implementation" section for a explanation of the implementation details.

&nbsp;
